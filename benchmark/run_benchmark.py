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
    approach6 — per-SQ hard routing (one distinct prompt per SQ)
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

from approach1_experiment import coords_to_world, compute_mesh_normalization, load_sq_params
from approach5_experiment import group_sqs_by_height, compute_hard_W, sample_slat_compositional
from approach6_experiment import compute_hard_W as compute_hard_W_6, sample_slat_extreme
from approach7_experiment import compute_soft_W as compute_soft_W_7, sample_slat_coupled


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def get_extrinsics_intrinsics():
    return render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2], [0.35] * 4, 10, 8
    )


def render_gaussian(pipeline, slat, extr, intr, prompt=""):
    gs = pipeline.decode_slat(slat, formats=["gaussian"])["gaussian"][0]
    scales = gs.get_scaling
    if not torch.isfinite(scales).all() or scales.max().item() > 10.0:
        print(f"  WARNING: degenerate Gaussian scales (max={scales.max().item():.3g}) — skipping render")
        del gs
        return None
    bg_color = (0, 0, 0) if "white" in prompt.lower() else (255, 255, 255)
    frames = render_utils.render_frames(
        gs, extr, intr, {"resolution": 512, "bg_color": bg_color}
    )["color"]
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
                  local_prompts, steps, seed, cfg_strength):
    W = compute_hard_W_6(coords_to_world(coords), sq_params, mesh_center, mesh_scale)
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts.items()}
    torch.manual_seed(seed)
    return sample_slat_extreme(
        pipeline, coords, W, conds_local,
        steps=steps, cfg_strength=cfg_strength,
    )


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

    if all(all_views_exist(output_root / f"prompt_{i}") for i in range(n_prompts)):
        print("  All renders already exist — skipping.")
        return

    sq_params = load_sq_params(os.path.join(results_root, shape["npz"]))
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)

    for prompt_idx in range(n_prompts):
        out_dir = output_root / f"prompt_{prompt_idx}"
        if all_views_exist(out_dir):
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

        if approach == "baseline":
            slat = run_baseline(pipeline, coords, global_prompt, steps, current_seed, cfg_strength)
        elif approach == "approach5":
            slat = run_approach5(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                 global_prompt, local_prompts, steps, current_seed, cfg_strength)
        elif approach == "approach6":
            slat = run_approach6(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                 local_prompts, steps, current_seed, cfg_strength)
        elif approach == "approach7":
            slat = run_approach7(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                 global_prompt, local_prompts, steps, current_seed, cfg_strength,
                                 lam=args.lam, tau=args.tau)

        frames = render_gaussian(pipeline, slat, extr, intr, prompt=global_prompt)
        del slat, coords, cond_struct
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
                        choices=["baseline", "approach5", "approach6", "approach7"])
    parser.add_argument("--shape-idx", default="all",
                        help="Index into prompts JSON (0-based), or 'all' to run every shape")
    parser.add_argument("--prompts-file", default="benchmark/prompts_augmented.json")
    parser.add_argument("--results-root", default=".")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-strength", type=float, default=7.5)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.02)
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
    pipeline = TrellisTextTo3DPipeline.from_pretrained(
        os.path.join(args.results_root, "gui")
    )
    pipeline.cuda()
    extr, intr = get_extrinsics_intrinsics()

    for idx in indices:
        run_shape(shapes[idx], args.approach, pipeline, extr, intr,
                  args.results_root, args.steps, args.seed, args.cfg_strength, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
