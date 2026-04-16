"""
Approach 1 diagnostic using Approach 6's per-SQ prompts.

Tests whether naive post-hoc feature transplant (Approach 1) can achieve
multi-material composition comparable to Approach 6's denoising-time routing.

For each unique prompt in extreme_prompts.txt, we generate a full SLAT
(same coords, same seed), then assemble a "composite" SLAT by picking each
voxel's features from the SLAT whose prompt matches that voxel's assigned SQ.

Outputs:
  - comparison_grid.png: global baseline | per-prompt SLATs | composite | approach6 (if found)
  - sq_assignment.png: voxel-to-SQ coloring
"""

import os
import sys
import torch
import numpy as np
from collections import OrderedDict
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)
os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

from approach1_experiment import (
    load_sq_params,
    compute_mesh_normalization,
    build_normalized_sq_mesh,
    coords_to_world,
    hard_assign_voxels,
    get_cameras,
    render_gs,
    render_sq_mesh_normals,
    save_comparison_grid,
    save_sq_assignment_viz,
)


# ---------------------------------------------------------------------------
# Parse the per-SQ prompt file produced by Approach 6
# ---------------------------------------------------------------------------

def load_extreme_prompts(path: str) -> dict:
    """
    Parse lines like 'SQ 0: a pink plastic chair leg' into {0: "a pink ..."}.
    """
    prompts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "SQ <idx>: <prompt>"
            prefix, prompt = line.split(":", 1)
            idx = int(prefix.strip().split()[1])
            prompts[idx] = prompt.strip()
    return prompts


def multi_source_transplant(slats: dict,
                            assignment: torch.Tensor,
                            sq_to_prompt: dict) -> sp.SparseTensor:
    """
    Build a composite SLAT: for each voxel, use features from the SLAT
    generated with the prompt assigned to that voxel's SQ.

    Args:
        slats: {prompt_string: SparseTensor} — one SLAT per unique prompt.
        assignment: (N,) int64 — SQ index for each voxel.
        sq_to_prompt: {sq_idx: prompt_string} — maps SQ → prompt.

    Returns:
        Composite SparseTensor with cherry-picked features.
    """
    # Use any SLAT as a template for coords / shape
    template = next(iter(slats.values()))
    composite_feats = template.feats.clone()

    for sq_idx, prompt in sq_to_prompt.items():
        mask = (assignment == sq_idx)
        if mask.any() and prompt in slats:
            composite_feats[mask] = slats[prompt].feats[mask]

    return sp.SparseTensor(feats=composite_feats, coords=template.coords)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    sq_path: str = "gui/superquadrics/chair_sq.npz",
    prompts_path: str = "approach6_results/extreme_prompts.txt",
    global_prompt: str = "a minimalist chair with four thin legs, crossbars, a seat cushion, and a backrest",
    seed: int = 42,
    t0_idx: int = 6,
    steps: int = 12,
    cfg_strength: float = 7.5,
    output_dir: str = "approach1_vs6_results",
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load SQ params + prompts -----------------------------------------
    print("Loading superquadric parameters...")
    sq_params = load_sq_params(sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)
    P = len(sq_params)

    sq_to_prompt = load_extreme_prompts(prompts_path)
    print(f"  {P} superquadrics, {len(sq_to_prompt)} prompts loaded")
    for idx in sorted(sq_to_prompt):
        print(f"    SQ {idx}: {sq_to_prompt[idx]}")

    # Unique prompts (deduplicate — e.g. multiple SQs share "a silver chair crossbar")
    unique_prompts = list(OrderedDict.fromkeys(sq_to_prompt.values()))
    print(f"  {len(unique_prompts)} unique prompts")

    # ---- Normalised SQ mesh -----------------------------------------------
    import open3d as o3d
    sq_mesh = build_normalized_sq_mesh(sq_params, mesh_center, mesh_scale)
    sq_mesh_path = os.path.join(output_dir, "sq_mesh.ply")
    o3d.io.write_triangle_mesh(sq_mesh_path, sq_mesh)

    # ---- Pipeline ---------------------------------------------------------
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    extrinsics, intrinsics = get_cameras()

    # ---- Stage 1: sample structure (spatial control) ----------------------
    print("Stage 1: sampling sparse structure...")
    cond_struct = pipeline.get_cond_text([global_prompt])
    torch.manual_seed(seed)
    spatial_latent = pipeline.encode_spatial_control(sq_mesh_path)
    cond_ctrl = {**cond_struct, 'control': spatial_latent}
    coords = pipeline.sample_sparse_structure(
        cond_ctrl, num_samples=1,
        sampler_params={
            "steps": steps,
            "cfg_strength": cfg_strength,
            "t0_idx_value": t0_idx,
            "spatial_control_mesh_path": sq_mesh_path,
        },
    )
    print(f"  {coords.shape[0]} active voxels")

    # ---- Hard voxel-to-SQ assignment --------------------------------------
    print("Computing voxel → SQ assignment...")
    voxel_pos = coords_to_world(coords).to(pipeline.device)
    assignment = hard_assign_voxels(voxel_pos, sq_params, mesh_center, mesh_scale)
    for i in range(P):
        n = (assignment == i).sum().item()
        label = sq_to_prompt.get(i, "<no prompt>")
        print(f"  SQ {i}: {n:>5} voxels  — {label}")

    # ---- Stage 2: generate one SLAT per unique prompt ---------------------
    slats = {}
    for idx, prompt in enumerate(unique_prompts):
        print(f"Stage 2 [{idx+1}/{len(unique_prompts)}]: sampling SLAT for '{prompt}'...")
        cond = pipeline.get_cond_text([prompt])
        torch.manual_seed(seed)
        slats[prompt] = pipeline.sample_slat(cond, coords)

    # ---- Also generate a global-prompt baseline SLAT ----------------------
    print(f"Stage 2: sampling global baseline SLAT ('{global_prompt}')...")
    cond_global = pipeline.get_cond_text([global_prompt])
    torch.manual_seed(seed)
    slat_global = pipeline.sample_slat(cond_global, coords)

    # ---- Build the composite SLAT -----------------------------------------
    print("Building composite SLAT (multi-source transplant)...")
    slat_composite = multi_source_transplant(slats, assignment, sq_to_prompt)

    # ---- Render all variants ----------------------------------------------
    rows, labels = [], []

    print("Rendering SQ mesh normals...")
    rows.append(render_sq_mesh_normals(sq_mesh, extrinsics, intrinsics))
    labels.append("Input superquadrics")

    print("Rendering global baseline...")
    rows.append(render_gs(pipeline, slat_global, extrinsics, intrinsics))
    labels.append(f"Global baseline — {global_prompt}")

    for prompt in unique_prompts:
        print(f"Rendering '{prompt}'...")
        rows.append(render_gs(pipeline, slats[prompt], extrinsics, intrinsics))
        labels.append(f"Full SLAT — {prompt}")

    print("Rendering composite (Approach 1 transplant)...")
    rows.append(render_gs(pipeline, slat_composite, extrinsics, intrinsics))
    labels.append("Composite (Approach 1 multi-source transplant)")

    # ---- Try to include Approach 6 result for comparison ------------------
    a6_img_path = os.path.join(project_root, "approach6_results", "extreme_6_materials_chair.png")
    if os.path.exists(a6_img_path):
        print("Loading Approach 6 result for comparison...")
        a6_img = np.array(Image.open(a6_img_path))
        # The approach 6 image is 4 views concatenated horizontally
        h = a6_img.shape[0]
        w_per = a6_img.shape[1] // 4
        a6_views = [a6_img[:, i*w_per:(i+1)*w_per] for i in range(4)]
        # Resize to match our render resolution if needed
        res = rows[0][0].shape[0]
        if a6_views[0].shape[0] != res:
            from PIL import Image as PILImage
            a6_views = [
                np.array(PILImage.fromarray(v).resize((res, res), PILImage.LANCZOS))
                for v in a6_views
            ]
        rows.append(a6_views)
        labels.append("Approach 6 result (denoising-time routing)")

    # ---- Save outputs -----------------------------------------------------
    print("Saving outputs...")
    save_comparison_grid(rows, labels, os.path.join(output_dir, "comparison_grid.png"))
    save_sq_assignment_viz(coords, assignment, P,
                           os.path.join(output_dir, "sq_assignment.png"))

    # ---- Per-SQ transplant grid (swap one SQ at a time) -------------------
    # Shows the effect of transplanting each SQ individually from its local
    # prompt SLAT into the global baseline — same diagnostic as original
    # Approach 1 but with the Approach 6 prompt set.
    print("\nRendering per-SQ transplant grid...")
    per_sq_rows = [rows[0], rows[1]]  # SQ mesh + global baseline
    per_sq_labels = [labels[0], labels[1]]

    for i in range(P):
        prompt = sq_to_prompt.get(i)
        if prompt is None or prompt not in slats:
            continue
        n_vox = (assignment == i).sum().item()
        print(f"  SQ {i} ({n_vox} voxels) — {prompt}")
        mask = (assignment == i)
        feats = slat_global.feats.clone()
        feats[mask] = slats[prompt].feats[mask]
        slat_mix = sp.SparseTensor(feats=feats, coords=slat_global.coords)
        per_sq_rows.append(render_gs(pipeline, slat_mix, extrinsics, intrinsics))
        per_sq_labels.append(f"SQ {i} → '{prompt}'  [{n_vox} vox]")

    save_comparison_grid(per_sq_rows, per_sq_labels,
                         os.path.join(output_dir, "per_sq_transplant_grid.png"))

    print(f"\nDone. Results in {output_dir}/")
    print("  comparison_grid.png       — all SLATs + composite + approach 6")
    print("  per_sq_transplant_grid.png — per-SQ transplant into global baseline")
    print("  sq_assignment.png         — voxel-to-SQ assignment")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Approach 1 diagnostic using Approach 6 per-SQ prompts")
    p.add_argument("--sq-path", default="gui/superquadrics/chair_sq.npz")
    p.add_argument("--prompts-path", default="approach6_results/extreme_prompts.txt")
    p.add_argument("--global-prompt",
                   default="a minimalist chair with four thin legs, crossbars, a seat cushion, and a backrest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t0-idx", type=int, default=6)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--cfg-strength", type=float, default=7.5)
    p.add_argument("--output-dir", default="approach1_vs6_results")
    args = p.parse_args()

    run(
        sq_path=args.sq_path,
        prompts_path=args.prompts_path,
        global_prompt=args.global_prompt,
        seed=args.seed,
        t0_idx=args.t0_idx,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        output_dir=args.output_dir,
    )
