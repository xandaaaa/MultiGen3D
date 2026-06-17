"""
benchmark/run_benchmark.py

Runs one approach on one or all shapes and saves renders in the layout
expected by benchmark/clip_score.py:

    <results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/view_<j>.png

Also saves a row image per prompt for quick visual inspection:

    <results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/grid.png

Approaches:
    baseline     — standard TRELLIS, global prompt only, text-driven structure (no SQ control)
    spacecontrol — SQ-mesh spatial control on the structure + single global-prompt SLAT
                   (the apples-to-apples reference for multigen: same geometry, no SQ routing)
    multigen     — compositional CFG via SQ region masks, with the sparse structure
                   conditioned on the merged SQ mesh via spatial control (matches the GUI)

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

from common.sq_utils import (
    coords_to_world,
    load_sq_params,
    save_sq_assignment_viz,
    compute_hard_W as compute_hard_W_sq,
    convert_shapenet_yup_to_trellis_zup,
)
from multigen import (
    compute_mesh_normalization,
    multigen_generate,
    sample_multigen_slat,
    write_spatial_control_mesh,
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def get_extrinsics_intrinsics():
    return render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2], [0.35] * 4, 10, 8
    )


def _render_gs(gs, extr, intr, prompt="", resolution=512):
    """Render an already-decoded Gaussian. Used by both the slat-based path
    (decode_slat → _render_gs) and the direct-Gaussian path (multigen)."""
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
                gs, extr_i, intr_i, {"resolution": resolution, "bg_color": bg_color}, verbose=False,
            )["color"]
        except torch.cuda.OutOfMemoryError as e:
            print(f"  WARNING: OOM on view {i}, retrying after cache clear: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            frames_i = render_utils.render_frames(
                gs, extr_i, intr_i, {"resolution": resolution, "bg_color": bg_color}, verbose=False,
            )["color"]
        frames.extend(frames_i)
        gc.collect()
        torch.cuda.empty_cache()
    return frames


def render_gaussian(pipeline, slat, extr, intr, prompt="", resolution=512):
    gs = pipeline.decode_slat(slat, formats=["gaussian"])["gaussian"][0]
    frames = _render_gs(gs, extr, intr, prompt=prompt, resolution=resolution)
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


def run_multigen(pipeline, coords, sq_params, mesh_center, mesh_scale,
              global_prompt, local_prompts, steps, seed, cfg_strength,
              local_cfg=15.0, soft_tau=None, debug_dir=None):
    """Compositional CFG: per-region CFG with shared noise trajectory.
    Returns a Gaussian directly unless debug_dir is set, in which case sampler
    snapshots are written and the final decode/render path is skipped.
    """
    cond_global = pipeline.get_cond_text([global_prompt])
    # Mirror the GUI (generate_local_sq): cover EVERY SQ, fall back to the global
    # prompt where a region has none, dedupe by prompt string, and reuse the
    # cond_global object for global-equal regions so they denoise at the global
    # cfg strength while others use local_cfg. Without this, every region runs at
    # local_cfg with no coherent global base, and any SQ missing from local_prompts
    # gets a zero mask and is left as undenoised noise.
    prompt_to_cond = {global_prompt: cond_global}
    conds_local = {}
    for i in range(len(sq_params)):
        p = local_prompts.get(i, "")
        p = (p.strip() if isinstance(p, str) else "") or global_prompt
        if p not in prompt_to_cond:
            prompt_to_cond[p] = pipeline.get_cond_text([p])
        conds_local[i] = prompt_to_cond[p]
    torch.manual_seed(seed)
    if debug_dir is not None:
        slat = sample_multigen_slat(
            pipeline, coords, conds_local, cond_global,
            sq_params, mesh_center, mesh_scale,
            steps=steps, cfg_strength=cfg_strength,
            local_cfg_strength=local_cfg, soft_tau=soft_tau,
            debug_dir=debug_dir,
        )
        del slat
        return None

    gs, _mesh = multigen_generate(
        pipeline, coords, conds_local, cond_global,
        sq_params, mesh_center, mesh_scale,
        steps=steps, cfg_strength=cfg_strength,
        local_cfg_strength=local_cfg, soft_tau=soft_tau,
    )
    return gs


# ---------------------------------------------------------------------------
# Per-shape runner
# ---------------------------------------------------------------------------

def run_shape(shape, approach, pipeline, extr, intr, results_root, steps, seed, cfg_strength, args):
    shape_id = shape["id"]
    n_prompts = len(shape["prompts"])
    out_subdir = f"{approach}_{args.output_suffix}_results" if args.output_suffix else f"{approach}_results"
    output_root = Path(results_root) / out_subdir / "renders" / shape_id

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
    if approach in ("multigen", "spacecontrol"):
        sq_params = convert_shapenet_yup_to_trellis_zup(sq_params)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)

    shape_debug_dir = None
    sq_viz_written = False
    if args.debug_dir and approach == "multigen":
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
        ss_sampler_params = {"steps": steps}
        if approach in ("multigen", "spacecontrol"):
            # Match the GUI: condition the sparse structure on the merged SQ mesh
            # so the voxels fill the superquadric volume. Without this the structure
            # comes from text alone and ignores the SQ layout, so the SLAT region
            # masks route over voxels that don't match the SQs. spacecontrol uses the
            # same SQ-locked geometry but a single global-prompt SLAT, so it is the
            # apples-to-apples reference for what multigen's per-region SLAT adds.
            # SQ artifacts (control mesh + the voxelized/latent debug plys that
            # encode_spatial_control drops next to it) go to a shared
            # results/superquadrics/<shape_id>/ instead of the render folders.
            sq_dir = Path(results_root) / "superquadrics" / shape_id
            sq_dir.mkdir(parents=True, exist_ok=True)
            control_path = str(sq_dir / "spatial_control_mesh.ply")
            write_spatial_control_mesh(sq_params, control_path, mesh_center, mesh_scale)
            cond_struct = {**cond_struct, "control": pipeline.encode_spatial_control(control_path)}
            # GUI uses t0_idx=6 at 12 structure steps (fraction 0.5). t0 depends
            # only on the idx/steps fraction, so scale with --steps to match.
            t0_idx = args.t0_idx if args.t0_idx is not None else round(0.5 * steps)
            ss_sampler_params.update(
                cfg_strength=cfg_strength, t0_idx_value=min(t0_idx, steps)
            )
        torch.manual_seed(current_seed)
        coords = pipeline.sample_sparse_structure(
            cond_struct, num_samples=1, sampler_params=ss_sampler_params
        )

        prompt_debug_dir = None
        if shape_debug_dir is not None:
            if not sq_viz_written:
                W_dbg = compute_hard_W_sq(
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
        if approach in ("baseline", "spacecontrol"):
            # Same single-global-prompt SLAT for both; they differ only in whether
            # `coords` came from SQ spatial control (spacecontrol) or text (baseline).
            slat = run_baseline(pipeline, coords, global_prompt, steps, current_seed, cfg_strength)
        elif approach == "multigen":
            gs_direct = run_multigen(
                pipeline, coords, sq_params, mesh_center, mesh_scale,
                global_prompt, local_prompts, steps, current_seed, cfg_strength,
                local_cfg=args.local_cfg, soft_tau=args.soft_tau,
                debug_dir=prompt_debug_dir,
            )
        else:
            sys.exit(f"Unknown approach: {approach!r}")

        if prompt_debug_dir is not None:
            # Diagnostic run: standard view renders already exist from prior
            # benchmark runs; skip the final render to leave GPU headroom for
            # decoded sampler snapshots on heavy shapes.
            if slat is not None: del slat
            if gs_direct is not None: del gs_direct
            del coords, cond_struct
            print(f"    [debug] skipped final render (diagnostic mode)")
        else:
            if approach == "multigen":
                # Multi-prompt-per-step samplers accumulate per-step GPU state;
                # clear before the rasterizer needs ~50MiB.
                gc.collect()
                torch.cuda.empty_cache()
            if gs_direct is not None:
                frames = _render_gs(gs_direct, extr, intr, prompt=global_prompt,
                                    resolution=args.resolution)
                del gs_direct
            else:
                frames = render_gaussian(pipeline, slat, extr, intr, prompt=global_prompt,
                                         resolution=args.resolution)
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
                        choices=["baseline", "spacecontrol", "multigen"])
    parser.add_argument("--shape-idx", default="all",
                        help="Index into prompts JSON (0-based), or 'all' to run every shape")
    parser.add_argument("--prompts-file", default="benchmark/prompts_augmented.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-strength", type=float, default=7.5)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate renders even when view PNGs already exist")
    parser.add_argument("--debug-dir", default=None,
                        help="If set (and approach=multigen), dump per-shape SQ assignment "
                             "viz and per-prompt decoded sampler snapshots.")
    parser.add_argument("--local-cfg", type=float, default=15.0,
                        help="multigen: per-region local-prompt CFG strength (default 15.0).")
    parser.add_argument("--soft-tau", type=float, default=None,
                        help="multigen: optional softmax temperature for soft SQ masks. "
                             "Omit for hard (one-hot) masks.")
    parser.add_argument("--t0-idx", type=int, default=None,
                        help="multigen: spatial-control strength as an index into the "
                             "sparse-structure t-schedule (higher = stronger adherence to "
                             "the SQ mesh). Default: round(0.5 * --steps), matching the GUI's "
                             "control slider (6 at 12 steps = fraction 0.5). "
                             "Clamped to --steps; only used when --approach multigen.")
    parser.add_argument("--output-suffix", default=None,
                        help="Optional suffix for the output directory, so several tuning runs "
                             "of the same approach don't collide. Example: 'v2' → results/"
                             "<approach>_v2_results/.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution in pixels (default 512 to match baseline; "
                             "drop to 256 if 16G GPUs OOM).")
    args = parser.parse_args()

    # Honor SLURM array semantics if shape-idx is left at its default ("all")
    # — that way a single array task in a job array picks up its own shape.
    if args.shape_idx == "all" and "SLURM_ARRAY_TASK_ID" in os.environ:
        slurm_id = os.environ["SLURM_ARRAY_TASK_ID"]
        # 4294967294 is the sentinel SLURM uses for a non-array job submission
        # that was started with sbatch's array env stub; ignore it.
        if slurm_id != "4294967294":
            args.shape_idx = slurm_id
            print(f"[run_benchmark] Using shape-idx={slurm_id} from SLURM_ARRAY_TASK_ID")

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
