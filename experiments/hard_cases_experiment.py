"""
Hard-case stress tests for Approach 1 (transplant) vs Approach 6 (per-SQ routing).

Tests prompts that go beyond solid colors: patterns, reflective materials,
geometric style conflicts, semantic context, and lighting mismatches.

For each test case, produces:
  - Baseline: vanilla TRELLIS with a single descriptive global prompt
  - Approach 1: independent SLATs (one per group), composited post-hoc
  - Approach 6-style: denoising-time routing with per-group prompts

Uses manual 4-group assignment (legs/crossbars/seat/backrest) derived from
the known chair_sq.npz layout.

Outputs per test case:
  - <case_name>_grid.png: baseline | approach 1 composite | approach 6 routing
"""

import os
import sys
import torch
import numpy as np
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont

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
)
from approach5_experiment import (
    compute_hard_W,
    sample_slat_compositional,
)


# ---------------------------------------------------------------------------
# Manual SQ → group mapping for chair_sq.npz
# Based on ground-truth part labels from approach6 extreme_prompts.txt:
#   Legs:      SQ 0, 1, 3, 5
#   Crossbars: SQ 2, 4, 6, 9
#   Seat:      SQ 8
#   Backrest:  SQ 7
# ---------------------------------------------------------------------------

CHAIR_GROUP_MAP = {
    0: 0, 1: 0, 3: 0, 5: 0,   # legs
    2: 1, 4: 1, 6: 1, 9: 1,   # crossbars
    8: 2,                       # seat
    7: 3,                       # backrest
}

GROUP_NAMES = {
    0: "Legs",
    1: "Crossbars",
    2: "Seat",
    3: "Backrest",
}

NUM_GROUPS = 4


# ---------------------------------------------------------------------------
# Test cases: each has a global prompt + 4 group prompts
# ---------------------------------------------------------------------------

TEST_CASES = OrderedDict({
    "patterns": {
        "description": "Patterns & textures spanning multiple voxels",
        "global": "a chair with striped black and white legs, silver crossbars, a plaid fabric seat, and a polka dot backrest",
        # Approach 1: full-chair prompts (generates entire SLAT per prompt)
        "a1_prompts": {
            0: "a chair with black and white vertically striped legs",
            1: "a chair with silver metallic crossbars",
            2: "a chair with a red and black plaid fabric seat",
            3: "a chair with a white and blue polka dot backrest",
        },
        # Approach 6: local material descriptors (routed to specific voxels)
        "a6_prompts": {
            0: "black and white vertically striped chair legs",
            1: "silver metallic crossbars",
            2: "a red and black plaid fabric seat cushion",
            3: "a white and blue polka dot backrest",
        },
    },
    "reflective": {
        "description": "View-dependent / reflective materials",
        "global": "a chair with chrome metallic legs, gold crossbars, a transparent glass seat, and a mirror-finish backrest",
        "a1_prompts": {
            0: "a chair with shiny chrome metallic legs",
            1: "a chair with gold metallic crossbars",
            2: "a chair with a transparent glass seat",
            3: "a chair with a mirror-finish reflective backrest",
        },
        "a6_prompts": {
            0: "shiny chrome metallic chair legs",
            1: "gold metallic crossbars",
            2: "a transparent glass seat",
            3: "a mirror-finish reflective backrest",
        },
    },
    "geometric_clash": {
        "description": "Conflicting geometric styles per region",
        "global": "a chair with rough jagged stone legs, thin iron crossbars, a smooth organic curved seat, and an ornate carved wooden backrest",
        "a1_prompts": {
            0: "a chair with rough jagged stone legs",
            1: "a chair with thin wrought iron crossbars",
            2: "a chair with a smooth organic curved cushioned seat",
            3: "a chair with an ornate carved wooden backrest with intricate details",
        },
        "a6_prompts": {
            0: "rough jagged stone chair legs",
            1: "thin wrought iron crossbars",
            2: "a smooth organic curved cushioned seat",
            3: "an ornate carved wooden backrest with intricate details",
        },
    },
    "semantic_context": {
        "description": "Prompts that imply cross-boundary phenomena",
        "global": "a chair with legs covered in green ivy vines, rusty metal crossbars, a mossy old stone seat, and a backrest made of twisted tree branches",
        "a1_prompts": {
            0: "a chair with legs covered in green ivy vines growing upward",
            1: "a chair with rusty corroded metal crossbars",
            2: "a chair with a mossy old stone seat with cracks",
            3: "a chair with a backrest made of twisted living tree branches",
        },
        "a6_prompts": {
            0: "chair legs covered in green ivy vines growing upward",
            1: "rusty corroded metal crossbars",
            2: "a mossy old stone seat with cracks",
            3: "a backrest made of twisted living tree branches",
        },
    },
    "lighting_conflict": {
        "description": "Conflicting lighting / emission assumptions",
        "global": "a chair with bright neon glowing pink legs, dark carbon fiber crossbars, a dark matte black rubber seat, and a translucent glowing blue backrest",
        "a1_prompts": {
            0: "a chair with bright neon glowing pink legs",
            1: "a chair with dark carbon fiber crossbars",
            2: "a chair with a dark matte black rubber seat",
            3: "a chair with a translucent glowing blue backrest",
        },
        "a6_prompts": {
            0: "bright neon glowing pink chair legs",
            1: "dark carbon fiber crossbars",
            2: "a dark matte black rubber seat",
            3: "a translucent glowing blue backrest",
        },
    },
})


# ---------------------------------------------------------------------------
# Approach 6-style routing at group level (3 groups, not P SQs)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_slat_group_routing(pipeline, coords, W, group_map,
                              conds_dict, n_groups, steps=15,
                              cfg_strength=7.5):
    """
    Denoising-time routing with N groups. Runs the flow model once per group
    per step, fuses velocities via spatial masks.

    Generalises approach5's sample_slat_compositional to any number of groups.
    """
    flow_model = pipeline.models['slat_flow_model_text']
    N, D = coords.shape[0], flow_model.in_channels
    device = pipeline.device

    # Build semantic mask: (N, n_groups)
    W_semantic = torch.zeros((N, n_groups), device=device)
    for sq_idx, group_idx in group_map.items():
        W_semantic[:, group_idx] += W[:, sq_idx]

    z_init = torch.randn(N, D, device=device)
    sample = sp.SparseTensor(feats=z_init, coords=coords)
    sampler = pipeline.slat_sampler
    t_seq = np.linspace(1, 0, steps + 1)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

    print(f"  Routing with {n_groups} groups, {steps} steps...")

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        dt = t_prev - t
        feats_fused = torch.zeros_like(sample.feats)

        for group_idx, cond in conds_dict.items():
            out = sampler.sample_once(
                flow_model, sample, t, t_prev, cond['cond'],
                cfg_strength=cfg_strength,
                neg_cond=cond.get('neg_cond'),
                cfg_interval=(0.0, 1.0),
            )
            mask = W_semantic[:, group_idx:group_idx + 1]
            feats_fused += mask * out.pred_x_prev.feats
            del out

        v_fused = (feats_fused - sample.feats) / dt
        sample = sample.replace(sample.feats + dt * v_fused)
        if step_idx % 3 == 0:
            print(f"    Step {step_idx}/{steps}")

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)
    return sample * std + mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    sq_path: str = "gui/superquadrics/chair_sq.npz",
    seed: int = 42,
    steps: int = 15,
    cfg_strength: float = 7.5,
    output_dir: str = "hard_cases_results",
    cases: str = "all",
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- SQ params + grouping ---------------------------------------------
    print("Loading superquadric parameters...")
    sq_params = load_sq_params(sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)
    P = len(sq_params)
    group_map = CHAIR_GROUP_MAP

    print(f"Manual grouping ({NUM_GROUPS} groups):")
    for g in range(NUM_GROUPS):
        sqs = [k for k, v in group_map.items() if v == g]
        print(f"  {GROUP_NAMES[g]}: SQs {sqs}")

    # ---- SQ mesh ----------------------------------------------------------
    import open3d as o3d
    sq_mesh = build_normalized_sq_mesh(sq_params, mesh_center, mesh_scale)
    sq_mesh_path = os.path.join(output_dir, "sq_mesh.ply")
    o3d.io.write_triangle_mesh(sq_mesh_path, sq_mesh)

    # ---- Pipeline ---------------------------------------------------------
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    extrinsics, intrinsics = get_cameras()

    # ---- Select cases to run ----------------------------------------------
    if cases == "all":
        selected = TEST_CASES
    else:
        selected = OrderedDict(
            (k, v) for k, v in TEST_CASES.items() if k in cases.split(","))

    # ---- Run each test case -----------------------------------------------
    for case_name, case in selected.items():
        print(f"\n{'='*60}")
        print(f"TEST CASE: {case_name}")
        print(f"  {case['description']}")
        print(f"{'='*60}")

        global_prompt = case["global"]
        a1_prompts = case["a1_prompts"]
        a6_prompts = case["a6_prompts"]

        print(f"  Global: {global_prompt}")
        for g in range(NUM_GROUPS):
            print(f"  {GROUP_NAMES[g]}:")
            print(f"    A1: {a1_prompts[g]}")
            print(f"    A6: {a6_prompts[g]}")

        # ---- Stage 1: structure -------------------------------------------
        print("\nStage 1: sampling sparse structure...")
        cond_global = pipeline.get_cond_text([global_prompt])
        torch.manual_seed(seed)
        coords = pipeline.sample_sparse_structure(
            cond_global, num_samples=1,
            sampler_params={"steps": steps},
        )
        print(f"  {coords.shape[0]} active voxels")

        # ---- Voxel → group assignment -------------------------------------
        voxel_pos = coords_to_world(coords).to(pipeline.device)
        W = compute_hard_W(voxel_pos, sq_params, mesh_center, mesh_scale)

        # Group assignment for Approach 1 transplant
        sq_assignment = W.argmax(dim=1)
        group_assignment = torch.zeros_like(sq_assignment)
        for sq_idx, group_idx in group_map.items():
            group_assignment[sq_assignment == sq_idx] = group_idx

        for g in range(NUM_GROUPS):
            print(f"  {GROUP_NAMES[g]}: {(group_assignment == g).sum().item()} voxels")

        # ---- Baseline: vanilla TRELLIS ------------------------------------
        print(f"\nBaseline: '{global_prompt}'")
        torch.manual_seed(seed)
        slat_baseline = pipeline.sample_slat(cond_global, coords)

        # ---- Approach 1: generate one SLAT per group + composite ----------
        slats_a1 = {}
        for g in range(NUM_GROUPS):
            prompt = a1_prompts[g]
            print(f"Approach 1 — SLAT for {GROUP_NAMES[g]}: '{prompt}'")
            cond = pipeline.get_cond_text([prompt])
            torch.manual_seed(seed)
            slats_a1[g] = pipeline.sample_slat(cond, coords)

        # Build composite
        composite_feats = slat_baseline.feats.clone()
        for g in range(NUM_GROUPS):
            mask = (group_assignment == g)
            composite_feats[mask] = slats_a1[g].feats[mask]
        slat_a1_composite = sp.SparseTensor(
            feats=composite_feats, coords=slat_baseline.coords)

        # ---- Approach 6-style: group-level denoising routing --------------
        print("Approach 6 — denoising-time group routing...")
        conds_local = {g: pipeline.get_cond_text([p])
                       for g, p in a6_prompts.items()}
        torch.manual_seed(seed)
        slat_routing = sample_slat_group_routing(
            pipeline, coords, W, group_map, conds_local,
            n_groups=NUM_GROUPS,
            steps=steps, cfg_strength=cfg_strength,
        )

        # ---- Render -------------------------------------------------------
        rows, labels = [], []

        print("Rendering SQ mesh...")
        rows.append(render_sq_mesh_normals(sq_mesh, extrinsics, intrinsics))
        labels.append("Input superquadrics")

        print("Rendering baseline...")
        rows.append(render_gs(pipeline, slat_baseline, extrinsics, intrinsics))
        labels.append(f"Baseline (vanilla TRELLIS) — {global_prompt}")

        # Render each Approach 1 intermediate SLAT (one per group prompt)
        for g in range(NUM_GROUPS):
            print(f"Rendering Approach 1 intermediate — {GROUP_NAMES[g]}...")
            rows.append(render_gs(pipeline, slats_a1[g], extrinsics, intrinsics))
            labels.append(f"A1 intermediate [{GROUP_NAMES[g]}] — {a1_prompts[g]}")

        print("Rendering Approach 1 composite...")
        rows.append(render_gs(pipeline, slat_a1_composite, extrinsics, intrinsics))
        labels.append("Approach 1 (post-hoc transplant)")

        print("Rendering denoising-time routing...")
        rows.append(render_gs(pipeline, slat_routing, extrinsics, intrinsics))
        labels.append("Approach 5/6 (denoising-time routing)")

        # ---- Save grid ----------------------------------------------------
        grid_path = os.path.join(output_dir, f"{case_name}_grid.png")
        save_comparison_grid(rows, labels, grid_path)

        # ---- Clean up VRAM ------------------------------------------------
        del slat_baseline, slat_a1_composite, slat_routing
        for s in slats_a1.values():
            del s
        del slats_a1
        torch.cuda.empty_cache()

    # ---- Save prompt summary ----------------------------------------------
    summary_path = os.path.join(output_dir, "test_cases.txt")
    with open(summary_path, "w") as f:
        for case_name, case in selected.items():
            f.write(f"=== {case_name}: {case['description']} ===\n")
            f.write(f"Global: {case['global']}\n")
            for g in range(NUM_GROUPS):
                f.write(f"  {GROUP_NAMES[g]}:\n")
                f.write(f"    A1: {case['a1_prompts'][g]}\n")
                f.write(f"    A6: {case['a6_prompts'][g]}\n")
            f.write("\n")

    print(f"\nDone. Results in {output_dir}/")
    for case_name in selected:
        print(f"  {case_name}_grid.png")
    print(f"  test_cases.txt")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Hard-case stress tests: Approach 1 vs 6")
    p.add_argument("--sq-path", default="gui/superquadrics/chair_sq.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--cfg-strength", type=float, default=7.5)
    p.add_argument("--output-dir", default="hard_cases_results")
    p.add_argument("--cases", default="all",
                   help="Comma-separated case names, or 'all'")
    args = p.parse_args()

    run(
        sq_path=args.sq_path,
        seed=args.seed,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        output_dir=args.output_dir,
        cases=args.cases,
    )