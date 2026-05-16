"""
benchmark/run_benchmark.py

Runs one approach on one or all shapes and saves renders in the layout
expected by benchmark/clip_score.py:

    <results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/view_<j>.png

Also saves a row image per prompt for quick visual inspection:

    <results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/grid.png

Approaches:
    baseline  — standard TRELLIS, global prompt only (no SQ routing)
    approach5 — height-grouped semantic routing (3 groups: bottom / mid / top)
    approach6 — legacy per-SQ hard routing (one distinct prompt per SQ)
    local_sq  — migrated local_sq.py implementation, saved under local_sq_results
    approach7 — coupled diffusion (soft W + global coupling branch)

Usage:
    python benchmark/run_benchmark.py --approach baseline --shape-idx 3
    python benchmark/run_benchmark.py --approach baseline --shape-idx all
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# --- Path setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "experiments"))
os.environ["SPCONV_ALGO"] = "native"

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils

from approach1_experiment import coords_to_world, compute_mesh_normalization, load_sq_params, save_sq_assignment_viz
from approach5_experiment import group_sqs_by_height, compute_hard_W, sample_slat_compositional
from approach6_experiment import compute_hard_W as compute_hard_W_6, sample_slat_regional_refine
from local_sq import (
    compute_hard_W as compute_hard_W_local_sq,
    convert_shapenet_yup_to_trellis_zup,
    make_contextual_local_prompts,
    sample_slat_extreme_v1 as sample_slat_local_sq_extreme_v1,
    sample_slat_regional_refine as sample_slat_local_sq_regional,
)
from approach7_experiment import compute_soft_W as compute_soft_W_7, sample_slat_coupled
from decode_composite import decode_composite_gaussian


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def get_extrinsics_intrinsics():
    return render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2], [0.35] * 4, 10, 8
    )


def _render_gs(gs, extr, intr, prompt=""):
    """Render an already-decoded Gaussian. Used by both the slat-based path
    (decode_slat → _render_gs) and the direct-Gaussian path (decode_composite)."""
    scales = gs.get_scaling
    if not torch.isfinite(scales).all() or scales.max().item() > 10.0:
        print(f"  WARNING: degenerate Gaussian scales (max={scales.max().item():.3g}) — skipping render")
        return None
    gc.collect()
    torch.cuda.empty_cache()
    bg_color = (0, 0, 0) if "white" in prompt.lower() else (255, 255, 255)
    # Render views one at a time with empty_cache between, because the rasterizer's
    # geomBuffer/binningBuffer/imgBuffer aren't released between consecutive renders
    # and we OOM on 16G GPUs after view 2 for heavy shapes (>12K voxels).
    n_views = len(extr) if isinstance(extr, list) else extr.shape[0]
    frames = []
    for i in range(n_views):
        extr_i = [extr[i]] if isinstance(extr, list) else extr[i:i+1]
        intr_i = [intr[i]] if isinstance(intr, list) else intr[i:i+1]
        try:
            frames_i = render_utils.render_frames(
                gs, extr_i, intr_i, {"resolution": 256, "bg_color": bg_color}, verbose=False,
            )["color"]
        except torch.cuda.OutOfMemoryError as e:
            print(f"  WARNING: OOM on view {i}, retrying after cache clear: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            frames_i = render_utils.render_frames(
                gs, extr_i, intr_i, {"resolution": 256, "bg_color": bg_color}, verbose=False,
            )["color"]
        frames.extend(frames_i)
        gc.collect()
        torch.cuda.empty_cache()
    return frames


def render_gaussian(pipeline, slat, extr, intr, prompt=""):
    gs = pipeline.decode_slat(slat, formats=["gaussian"])["gaussian"][0]
    frames = _render_gs(gs, extr, intr, prompt=prompt)
    del gs
    return frames


def save_renders(frames, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, frame in enumerate(frames):
        Image.fromarray(frame).save(out_dir / f"view_{j}.png")


def all_views_exist(out_dir: Path, n_views: int = 4) -> bool:
    return all((out_dir / f"view_{j}.png").exists() for j in range(n_views))


# ---------------------------------------------------------------------------
# Per-approach runners  (all return a raw slat tensor)
# ---------------------------------------------------------------------------

def run_baseline(pipeline, coords, global_prompt, steps, seed, cfg_strength):
    cond = pipeline.get_cond_text([global_prompt])
    torch.manual_seed(seed)
    return pipeline.sample_slat(cond, coords, sampler_params={"steps": steps})


def run_approach5(pipeline, coords, sq_params, mesh_center, mesh_scale,
                  global_prompt, local_prompts, steps, seed, cfg_strength):
    W = compute_hard_W(coords_to_world(coords), sq_params, mesh_center, mesh_scale)
    group_map = group_sqs_by_height(sq_params, mesh_center)

    # One prompt per height group — take the first SQ assigned to each group
    group_prompts = {}
    for sq_idx, grp in group_map.items():
        if grp not in group_prompts:
            group_prompts[grp] = local_prompts.get(sq_idx, global_prompt)
    for g in range(3):
        if g not in group_prompts:
            group_prompts[g] = global_prompt

    conds_local = {g: pipeline.get_cond_text([p]) for g, p in group_prompts.items()}
    torch.manual_seed(seed)
    return sample_slat_compositional(
        pipeline, coords, W, group_map, conds_local,
        steps=steps, cfg_strength=cfg_strength,
    )


def run_approach6(pipeline, coords, sq_params, mesh_center, mesh_scale,
                  global_prompt, local_prompts, steps, seed, cfg_strength):
    W = compute_hard_W_6(coords_to_world(coords), sq_params, mesh_center, mesh_scale)
    cond_global = pipeline.get_cond_text([global_prompt])
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts.items()}
    torch.manual_seed(seed)
    return sample_slat_regional_refine(
        pipeline, coords, W, conds_local, cond_global,
        global_steps=steps, cfg_strength=cfg_strength,
    )


def run_local_sq(pipeline, coords, sq_params, mesh_center, mesh_scale,
                 global_prompt, local_prompts, steps, seed, cfg_strength,
                 debug_dir=None, structural_global_prompt=None,
                 sampler_variant="extreme_v1"):
    W = compute_hard_W_local_sq(coords_to_world(coords), sq_params, mesh_center, mesh_scale)
    contextual_prompts = make_contextual_local_prompts(global_prompt, local_prompts)
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in contextual_prompts.items()}
    torch.manual_seed(seed)

    if sampler_variant == "extreme_v1":
        # Pre-91da279 multi-prompt fusion: the variant that empirically achieves
        # distinct per-SQ colors on the chair test. No global cond, CFG always on.
        return sample_slat_local_sq_extreme_v1(
            pipeline, coords, W, conds_local,
            steps=steps, cfg_strength=cfg_strength,
            debug_dir=debug_dir,
        )

    if sampler_variant == "regional_refine":
        # Two-stage variant: global denoising + per-SQ partial refinement.
        # Empirically doesn't move colors much beyond x0_global's defaults.
        stage1_prompt = structural_global_prompt or global_prompt
        cond_global = pipeline.get_cond_text([stage1_prompt])
        return sample_slat_local_sq_regional(
            pipeline, coords, W, conds_local, cond_global,
            global_steps=steps, cfg_strength=cfg_strength,
            debug_dir=debug_dir,
        )

    raise ValueError(f"Unknown sampler_variant: {sampler_variant!r}")


def run_decode_composite(pipeline, coords, sq_params, mesh_center, mesh_scale,
                         global_prompt, local_prompts, steps, seed, cfg_strength,
                         local_cfg=15.0, soft_tau=None):
    """Compositional CFG: per-region CFG with shared noise trajectory.
    Returns a Gaussian directly (decoder runs inside decode_composite_gaussian).
    """
    cond_global = pipeline.get_cond_text([global_prompt])
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts.items()}
    torch.manual_seed(seed)
    gs, _mesh = decode_composite_gaussian(
        pipeline, coords, conds_local, cond_global,
        sq_params, mesh_center, mesh_scale,
        steps=steps, cfg_strength=cfg_strength,
        local_cfg_strength=local_cfg, soft_tau=soft_tau,
    )
    return gs


def run_approach7(pipeline, coords, sq_params, mesh_center, mesh_scale,
                  global_prompt, local_prompts, steps, seed, cfg_strength,
                  lam=0.3, tau=0.02):
    W = compute_soft_W_7(coords_to_world(coords), sq_params, mesh_center, mesh_scale, tau=tau)
    cond_global = pipeline.get_cond_text([global_prompt])
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts.items()}
    torch.manual_seed(seed)
    return sample_slat_coupled(
        pipeline, coords, W, conds_local, cond_global,
        steps=steps, cfg_strength=cfg_strength, lam=lam,
    )


# ---------------------------------------------------------------------------
# Per-shape runner
# ---------------------------------------------------------------------------

def run_shape(shape, approach, pipeline, extr, intr, results_root, steps, seed, cfg_strength, args):
    shape_id = shape["id"]
    n_prompts = len(shape["prompts"])
    output_root = Path(results_root) / f"{approach}_results" / "renders" / shape_id

    print(f"\n{'='*60}")
    print(f"Shape: {shape_id}  ({n_prompts} prompts)")

    if not args.force and all(all_views_exist(output_root / f"prompt_{i}") for i in range(n_prompts)):
        print("  All renders already exist — skipping.")
        return

    sq_params = load_sq_params(os.path.join(project_root, shape["npz"]))
    # superdec npz files are Y-up (ShapeNet convention); TRELLIS sparse coords
    # are Z-up. Approaches that route voxels to SQs by radial distance are
    # frame-sensitive — convert here so the routing math operates in the same
    # frame as the voxels. Other approaches retain prior behavior.
    if approach in ("local_sq", "decode_composite"):
        sq_params = convert_shapenet_yup_to_trellis_zup(sq_params)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)

    shape_debug_dir = None
    sq_viz_written = False
    if args.debug_dir and approach == "local_sq":
        shape_debug_dir = Path(args.debug_dir) / shape_id
        shape_debug_dir.mkdir(parents=True, exist_ok=True)

    for prompt_idx in range(n_prompts):
        out_dir = output_root / f"prompt_{prompt_idx}"
        if not args.force and all_views_exist(out_dir):
            print(f"  prompt_{prompt_idx}: already rendered, skipping")
            continue

        global_prompt = shape["prompts"][prompt_idx]
        raw_local = shape.get("local_prompts", [{}] * n_prompts)[prompt_idx]
        local_prompts = {int(k): v for k, v in raw_local.items()}
        per_prompt_seed = (shape.get("seeds") or [None] * n_prompts)[prompt_idx]
        current_seed = per_prompt_seed if per_prompt_seed is not None else seed

        print(f"\n  prompt_{prompt_idx}: {global_prompt[:80]}")
        if per_prompt_seed is not None:
            print(f"  (using per-prompt seed={current_seed})")

        cond_struct = pipeline.get_cond_text([global_prompt])
        torch.manual_seed(current_seed)
        coords = pipeline.sample_sparse_structure(
            cond_struct, num_samples=1, sampler_params={"steps": steps}
        )

        prompt_debug_dir = None
        if shape_debug_dir is not None:
            if not sq_viz_written:
                W_dbg = compute_hard_W_local_sq(
                    coords_to_world(coords), sq_params, mesh_center, mesh_scale
                )
                P_dbg = W_dbg.shape[1]
                assignment = W_dbg.argmax(dim=1)
                save_sq_assignment_viz(
                    coords, assignment, n_sqs=P_dbg,
                    output_path=str(shape_debug_dir / "sq_assignment.png"),
                    panel_titles=('Top  (XY)', 'Side  (XZ)', 'Front  (YZ)'),
                )
                stats = {
                    "n_voxels_total": int(coords.shape[0]),
                    "per_sq_active_voxels": {
                        int(i): int((W_dbg[:, i] > 0.02).sum().item())
                        for i in range(P_dbg)
                    },
                    "mask_threshold": 0.02,
                }
                with open(shape_debug_dir / "sq_stats.json", "w") as f:
                    json.dump(stats, f, indent=2)
                print(f"  [debug] wrote {shape_debug_dir/'sq_assignment.png'}")
                print(f"  [debug] wrote {shape_debug_dir/'sq_stats.json'}: {stats}")
                sq_viz_written = True
            prompt_debug_dir = str(shape_debug_dir / f"prompt_{prompt_idx}")

        slat = None
        gs_direct = None
        if approach == "baseline":
            slat = run_baseline(pipeline, coords, global_prompt, steps, current_seed, cfg_strength)
        elif approach == "approach5":
            slat = run_approach5(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                 global_prompt, local_prompts, steps, current_seed, cfg_strength)
        elif approach == "approach6":
            slat = run_approach6(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                 global_prompt, local_prompts, steps, current_seed, cfg_strength)
        elif approach == "local_sq":
            slat = run_local_sq(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                global_prompt, local_prompts, steps, current_seed, cfg_strength,
                                debug_dir=prompt_debug_dir,
                                structural_global_prompt=shape.get("global_description"),
                                sampler_variant=args.sampler_variant)
        elif approach == "approach7":
            slat = run_approach7(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                 global_prompt, local_prompts, steps, current_seed, cfg_strength,
                                 lam=args.lam, tau=args.tau)
        elif approach == "decode_composite":
            # Returns a Gaussian directly (sampler decodes internally).
            gs_direct = run_decode_composite(
                pipeline, coords, sq_params, mesh_center, mesh_scale,
                global_prompt, local_prompts, steps, current_seed, cfg_strength,
                local_cfg=args.local_cfg, soft_tau=args.soft_tau,
            )
        else:
            sys.exit(f"Unknown approach: {approach!r}")

        if prompt_debug_dir is not None:
            # Diagnostic run: standard view renders already exist from prior
            # benchmark runs; skip the final render to leave GPU headroom for
            # per-SQ snapshots on heavy shapes.
            if slat is not None: del slat
            if gs_direct is not None: del gs_direct
            del coords, cond_struct
            print(f"    [debug] skipped final render (diagnostic mode)")
        else:
            if approach in ("local_sq", "decode_composite"):
                # Multi-prompt-per-step samplers accumulate per-step GPU state;
                # clear before the rasterizer needs ~50MiB.
                gc.collect()
                torch.cuda.empty_cache()
            if gs_direct is not None:
                frames = _render_gs(gs_direct, extr, intr, prompt=global_prompt)
                del gs_direct
            else:
                frames = render_gaussian(pipeline, slat, extr, intr, prompt=global_prompt)
                del slat
            del coords, cond_struct
            if frames is None:
                print(f"    WARNING: degenerate Gaussian for prompt_{prompt_idx}, skipping")
            else:
                save_renders(frames, out_dir)
                print(f"    Saved {len(frames)} views → {out_dir}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"Shape {shape_id} done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", required=True,
                        choices=["baseline", "approach5", "approach6", "local_sq", "approach7",
                                 "decode_composite"])
    parser.add_argument("--shape-idx", default="all",
                        help="Index into prompts JSON (0-based), or 'all' to run every shape")
    parser.add_argument("--prompts-file", default="benchmark/prompts_augmented.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-strength", type=float, default=7.5)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate renders even when view PNGs already exist")
    parser.add_argument("--debug-dir", default=None,
                        help="If set (and approach=local_sq), dump per-shape SQ assignment "
                             "viz and per-prompt decoded snapshots after each refinement stage.")
    parser.add_argument("--sampler-variant", default="extreme_v1",
                        choices=["extreme_v1", "regional_refine"],
                        help="local_sq sampler. 'extreme_v1' (default): pre-91da279 multi-prompt "
                             "fusion (achieves visible per-SQ coloring). 'regional_refine': "
                             "two-stage global+refinement (weak color control, kept for ablation).")
    parser.add_argument("--local-cfg", type=float, default=15.0,
                        help="decode_composite: per-region local-prompt CFG strength (default 15.0).")
    parser.add_argument("--soft-tau", type=float, default=None,
                        help="decode_composite: optional softmax temperature for soft SQ masks. "
                             "Omit for hard (one-hot) masks.")
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        shapes = json.load(f)

    if args.shape_idx == "all":
        indices = list(range(len(shapes)))
    else:
        idx = int(args.shape_idx)
        if idx >= len(shapes):
            sys.exit(f"shape-idx {idx} out of range (have {len(shapes)} shapes)")
        indices = [idx]

    print(f"Approach : {args.approach}")
    print(f"Shapes   : {indices}")

    print("Loading pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(os.path.join(project_root, "gui"))
    pipeline.cuda()
    extr, intr = get_extrinsics_intrinsics()

    for idx in indices:
        run_shape(shapes[idx], args.approach, pipeline, extr, intr,
                  args.results_root, args.steps, args.seed, args.cfg_strength, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
