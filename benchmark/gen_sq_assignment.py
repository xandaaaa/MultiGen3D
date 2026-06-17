"""Generate per-shape SQ assignment visualizations (sq_assignment.png + sq_stats.json)
for every shape in prompts_augmented.json, using TRELLIS-generated sparse coords
(NOT a synthetic dense grid). This requires loading TRELLIS and running its sparse
structure sampler — fast (~5-10s/shape) since we skip the SLAT denoise.

Per-shape coords are sampled using the first prompt; the routing rule itself is
deterministic given coords + SQ params, so the viz is representative.

Usage (must run on a GPU node, e.g. via sbatch):
    python benchmark/gen_sq_assignment.py
    python benchmark/gen_sq_assignment.py --out-dir results/local_sq_diagnostics
    python benchmark/gen_sq_assignment.py --shape-idx 19    # watercraft only

Outputs:
    <out-dir>/<shape_id>/sq_assignment.png
    <out-dir>/<shape_id>/sq_stats.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "experiments"))
os.environ["SPCONV_ALGO"] = "native"

from trellis.pipelines import TrellisTextTo3DPipeline

from common.sq_utils import (
    coords_to_world,
    load_sq_params,
    save_sq_assignment_viz,
    compute_hard_W,
    convert_shapenet_yup_to_trellis_zup,
)
from multigen import compute_mesh_normalization


MASK_THRESHOLD = 0.02


def process_shape(pipeline, shape: dict, out_root: Path, steps: int, seed: int) -> None:
    shape_id = shape["id"]
    npz_path = os.path.join(project_root, shape["npz"])
    if not os.path.exists(npz_path):
        print(f"  SKIP {shape_id}: missing npz {npz_path}")
        return

    sq_params = load_sq_params(npz_path)
    sq_params = convert_shapenet_yup_to_trellis_zup(sq_params)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)

    # Sample sparse structure using the first prompt (any prompt would work; the
    # routing rule is voxel-position-based so the partition is determined by
    # whichever coord set we get back).
    global_prompt = shape["prompts"][0]
    per_prompt_seed = (shape.get("seeds") or [None] * len(shape["prompts"]))[0]
    current_seed = per_prompt_seed if per_prompt_seed is not None else seed

    cond_struct = pipeline.get_cond_text([global_prompt])
    torch.manual_seed(current_seed)
    coords = pipeline.sample_sparse_structure(
        cond_struct, num_samples=1, sampler_params={"steps": steps}
    )

    W = compute_hard_W(coords_to_world(coords), sq_params, mesh_center, mesh_scale)
    P = W.shape[1]
    assignment = W.argmax(dim=1)

    shape_out = out_root / shape_id
    shape_out.mkdir(parents=True, exist_ok=True)

    save_sq_assignment_viz(
        coords, assignment, n_sqs=P,
        output_path=str(shape_out / "sq_assignment.png"),
        panel_titles=("Top  (XY)", "Side  (XZ)", "Front  (YZ)"),
    )

    stats = {
        "n_voxels_total": int(coords.shape[0]),
        "per_sq_active_voxels": {
            int(i): int((W[:, i] > MASK_THRESHOLD).sum().item())
            for i in range(P)
        },
        "mask_threshold": MASK_THRESHOLD,
        "seed": int(current_seed),
        "prompt_used_for_structure": global_prompt,
    }
    with open(shape_out / "sq_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {shape_out/'sq_assignment.png'}  ({P} SQs, {coords.shape[0]} voxels)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file",
                        default=os.path.join(project_root, "benchmark/prompts_augmented.json"))
    parser.add_argument("--out-dir",
                        default=os.path.join(project_root, "results/local_sq_diagnostics"))
    parser.add_argument("--shape-idx", default="all",
                        help="Index into prompts JSON (0-based), or 'all' (default).")
    parser.add_argument("--steps", type=int, default=15,
                        help="Sparse-structure sampler steps (default 15, matches run_benchmark).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        shapes = json.load(f)

    if args.shape_idx == "all":
        indices = list(range(len(shapes)))
    else:
        indices = [int(args.shape_idx)]

    out_root = Path(args.out_dir)
    print(f"Writing to {out_root}")

    print("Loading TRELLIS pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(os.path.join(project_root, "gui"))
    pipeline.cuda()

    for i in indices:
        print(f"\nShape {i}: {shapes[i]['id']}")
        process_shape(pipeline, shapes[i], out_root, args.steps, args.seed)


if __name__ == "__main__":
    main()
