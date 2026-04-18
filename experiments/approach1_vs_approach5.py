"""
Approach 1 diagnostic using Approach 5's 3-group prompts.

Tests whether naive post-hoc feature transplant at the semantic-group level
(Bottom/Middle/Top) can match Approach 5's denoising-time spatial routing.

For each of the 3 groups we generate a full SLAT (same coords, same seed),
then assemble a composite where each voxel gets features from the SLAT
whose prompt matches its height-based group.

Outputs:
  - comparison_grid.png: SQ mesh | global baseline | per-group SLATs |
                         composite | approach 5 result (if found)
  - per_group_transplant_grid.png: one-group-at-a-time swaps into baseline
  - sq_assignment.png: voxel-to-SQ colouring
"""

import os
import sys
import torch
import numpy as np
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
    get_cameras,
    render_gs,
    render_sq_mesh_normals,
    save_comparison_grid,
    save_sq_assignment_viz,
)
from approach5_experiment import (
    compute_hard_W,
    group_sqs_by_height,
)


GROUP_NAMES = {0: "Bottom (legs)", 1: "Middle (seat)", 2: "Top (backrest)"}


def compute_group_assignment(W: torch.Tensor,
                             group_map: dict) -> torch.Tensor:
    """
    Collapse per-SQ hard W into a per-voxel group label (0/1/2).

    Args:
        W: (N, P) one-hot SQ assignment matrix.
        group_map: {sq_idx: group_idx}.

    Returns:
        (N,) int64 tensor of group indices.
    """
    sq_assignment = W.argmax(dim=1)  # (N,) which SQ each voxel belongs to
    group_assignment = torch.zeros_like(sq_assignment)
    for sq_idx, group_idx in group_map.items():
        group_assignment[sq_assignment == sq_idx] = group_idx
    return group_assignment


def run(
    sq_path: str = "gui/superquadrics/chair_sq.npz",
    global_prompt: str = "a plastic chair, strictly solid red seat, solid blue backrest, and solid yellow legs",
    seed: int = 42,
    steps: int = 15,
    cfg_strength: float = 7.5,
    output_dir: str = "approach1_vs5_results",
):
    # Use the same prompts as Approach 5
    local_prompts = {
        0: "a plastic chair with solid yellow legs",
        1: "a plastic chair with a solid red seat",
        2: "a plastic chair with a solid blue backrest",
    }

    os.makedirs(output_dir, exist_ok=True)

    # ---- SQ params + grouping ---------------------------------------------
    print("Loading superquadric parameters...")
    sq_params = load_sq_params(sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)
    P = len(sq_params)
    group_map = group_sqs_by_height(sq_params, mesh_center)

    print(f"  {P} superquadrics")
    for g in range(3):
        sqs = [k for k, v in group_map.items() if v == g]
        print(f"  {GROUP_NAMES[g]}: SQs {sqs} — '{local_prompts[g]}'")

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

    # ---- Stage 1: structure -----------------------------------------------
    print("Stage 1: sampling sparse structure...")
    cond_global = pipeline.get_cond_text([global_prompt])
    torch.manual_seed(seed)
    coords = pipeline.sample_sparse_structure(
        cond_global, num_samples=1,
        sampler_params={"steps": steps},
    )
    print(f"  {coords.shape[0]} active voxels")

    # ---- Voxel → group assignment -----------------------------------------
    print("Computing voxel → group assignment...")
    voxel_pos = coords_to_world(coords).to(pipeline.device)
    W = compute_hard_W(voxel_pos, sq_params, mesh_center, mesh_scale)
    group_assignment = compute_group_assignment(W, group_map)
    for g in range(3):
        print(f"  {GROUP_NAMES[g]}: {(group_assignment == g).sum().item()} voxels")

    # ---- Stage 2: one SLAT per group prompt + global baseline -------------
    slats = {}
    for g in range(3):
        prompt = local_prompts[g]
        print(f"Stage 2: sampling SLAT for {GROUP_NAMES[g]} ('{prompt}')...")
        cond = pipeline.get_cond_text([prompt])
        torch.manual_seed(seed)
        slats[g] = pipeline.sample_slat(cond, coords)

    print(f"Stage 2: sampling global baseline ('{global_prompt}')...")
    torch.manual_seed(seed)
    slat_global = pipeline.sample_slat(cond_global, coords)

    # ---- Build composite SLAT ---------------------------------------------
    print("Building composite SLAT (group-level transplant)...")
    composite_feats = slat_global.feats.clone()
    for g in range(3):
        mask = (group_assignment == g)
        composite_feats[mask] = slats[g].feats[mask]
    slat_composite = sp.SparseTensor(feats=composite_feats, coords=slat_global.coords)

    # ---- Render all variants ----------------------------------------------
    rows, labels = [], []

    print("Rendering SQ mesh normals...")
    rows.append(render_sq_mesh_normals(sq_mesh, extrinsics, intrinsics))
    labels.append("Input superquadrics")

    print("Rendering global baseline...")
    rows.append(render_gs(pipeline, slat_global, extrinsics, intrinsics))
    labels.append(f"Global baseline — {global_prompt}")

    for g in range(3):
        print(f"Rendering {GROUP_NAMES[g]} SLAT...")
        rows.append(render_gs(pipeline, slats[g], extrinsics, intrinsics))
        labels.append(f"Full SLAT — {local_prompts[g]}")

    print("Rendering composite (Approach 1 group transplant)...")
    rows.append(render_gs(pipeline, slat_composite, extrinsics, intrinsics))
    labels.append("Composite (Approach 1 group-level transplant)")

    # ---- Try to include Approach 5 result for comparison ------------------
    a5_img_path = os.path.join(project_root, "approach5_results",
                               "exp5_vs_baseline_color_routing.png")
    if os.path.exists(a5_img_path):
        print("Loading Approach 5 result for comparison...")
        a5_full = np.array(Image.open(a5_img_path))
        # Approach 5 grid: 2 rows (baseline, exp5), each with 4 views.
        # The exp5 row is the bottom half (below the header).
        h_full = a5_full.shape[0]
        # Take the bottom row (exp5 result), skip header
        row_h = h_full // 2
        a5_row = a5_full[row_h:, :]
        # Strip the ~100px text header from this row
        header_px = 100
        a5_row = a5_row[header_px:, :]
        w_per = a5_row.shape[1] // 4
        a5_h = a5_row.shape[0]
        a5_views = [a5_row[:, i*w_per:(i+1)*w_per] for i in range(4)]
        # Resize to match our render resolution
        res = rows[0][0].shape[0]
        if a5_views[0].shape[0] != res:
            a5_views = [
                np.array(Image.fromarray(v).resize((res, res), Image.LANCZOS))
                for v in a5_views
            ]
        rows.append(a5_views)
        labels.append("Approach 5 result (denoising-time routing)")

    # ---- Save main grid ---------------------------------------------------
    print("Saving outputs...")
    save_comparison_grid(rows, labels, os.path.join(output_dir, "comparison_grid.png"))

    # ---- Per-group transplant grid (one group at a time) ------------------
    print("\nRendering per-group transplant grid...")
    pg_rows = [rows[0], rows[1]]  # SQ mesh + global baseline
    pg_labels = [labels[0], labels[1]]

    for g in range(3):
        n_vox = (group_assignment == g).sum().item()
        print(f"  {GROUP_NAMES[g]} ({n_vox} voxels)...")
        mask = (group_assignment == g)
        feats = slat_global.feats.clone()
        feats[mask] = slats[g].feats[mask]
        slat_mix = sp.SparseTensor(feats=feats, coords=slat_global.coords)
        pg_rows.append(render_gs(pipeline, slat_mix, extrinsics, intrinsics))
        pg_labels.append(f"{GROUP_NAMES[g]} → '{local_prompts[g]}'  [{n_vox} vox]")

    pg_rows.append(rows[-2] if not os.path.exists(a5_img_path) else rows[-2])
    pg_labels.append("Composite (all 3 groups transplanted)")

    save_comparison_grid(pg_rows, pg_labels,
                         os.path.join(output_dir, "per_group_transplant_grid.png"))

    # ---- SQ assignment viz (coloured by group) ----------------------------
    # Re-use save_sq_assignment_viz but with group labels instead of SQ labels
    save_sq_assignment_viz(coords, group_assignment, 3,
                           os.path.join(output_dir, "group_assignment.png"))

    print(f"\nDone. Results in {output_dir}/")
    print("  comparison_grid.png          — all SLATs + composite + approach 5")
    print("  per_group_transplant_grid.png — per-group transplant into baseline")
    print("  group_assignment.png         — voxel-to-group colouring")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Approach 1 diagnostic using Approach 5 group prompts")
    p.add_argument("--sq-path", default="gui/superquadrics/chair_sq.npz")
    p.add_argument("--global-prompt",
                   default="a plastic chair, strictly solid red seat, solid blue backrest, and solid yellow legs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--cfg-strength", type=float, default=7.5)
    p.add_argument("--output-dir", default="approach1_vs5_results")
    args = p.parse_args()

    run(
        sq_path=args.sq_path,
        global_prompt=args.global_prompt,
        seed=args.seed,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        output_dir=args.output_dir,
    )
