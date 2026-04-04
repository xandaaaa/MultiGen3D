"""
Approach 2: P-Noise Initialization with Superquadric Projection

Samples P noise vectors (one per superquadric), broadcasts them
to N voxels via the weight matrix W, then runs the pretrained TRELLIS
Stage 2 transformer with projection back to P features at each step.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Tuple, Optional

os.environ['SPCONV_ALGO'] = 'native'
sys.path.insert(0, os.path.dirname(__file__))

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

from approach1_experiment import (
    compute_weight_matrix,
    project_onto_sq_subspace,
    coords_to_world_positions,
    compute_mesh_normalization,
    load_sq_params,
    create_sq_assignment_viz,
    sample_slat_with_projection,
)


# ---------------------------------------------------------------------------
# Core: Supervisor's pipeline sampling function
# ---------------------------------------------------------------------------

def sample_slat_p_noise(
    pipeline: TrellisTextTo3DPipeline,
    cond: dict,
    coords: torch.Tensor,
    W: torch.Tensor,
    sampler_params: dict = {},
    project_every: int = 1,
    blend_alpha: float = 1.0,
    project_after_frac: float = 0.0,
    rescale_noise: bool = True,
    final_project: bool = True,
) -> sp.SparseTensor:
    """
    Approach 2: sample P noise vectors, broadcast to N voxels,
    denoise with pretrained TRELLIS, project back to P features, repeat.

    Args:
        pipeline: the TRELLIS pipeline.
        cond: conditioning dict with 'cond' and 'neg_cond'.
        coords: (N, 4) voxel coordinates.
        W: (N, P) precomputed weight matrix.
        sampler_params: sampler parameters.
        project_every: project every k steps (1 = every step).
        blend_alpha: mixing ratio in [0, 1]. 1 = full projection.
        project_after_frac: only start projecting after this fraction of steps.
        rescale_noise: if True, rescale broadcast noise to unit variance per voxel.

    Returns:
        slat: SparseTensor with denoised latents.
    """
    # Select flow model based on conditioning type
    if cond['cond'].shape[-1] == 768:
        flow_model = pipeline.models['slat_flow_model_text']
    else:
        flow_model = pipeline.models['slat_flow_model_image']

    N = coords.shape[0]
    P = W.shape[1]
    D = flow_model.in_channels  # 8

    # --- KEY DIFFERENCE: Sample P noise vectors and broadcast ---
    s_noise = torch.randn(P, D, device=pipeline.device)
    z_init = W @ s_noise  # (N, D)

    if rescale_noise:
        # Broadcasting through W reduces variance: Var(z_j) = sum_i w_ij^2 < 1
        # Rescale each voxel to restore unit variance
        row_sq_norm = (W ** 2).sum(dim=1).sqrt().unsqueeze(1)  # (N, 1)
        z_init = z_init / row_sq_norm.clamp(min=1e-8)

    sample = sp.SparseTensor(feats=z_init, coords=coords)

    # --- Standard denoising setup ---
    params = {**pipeline.slat_sampler_params, **sampler_params}
    sampler = pipeline.slat_sampler

    steps = params.get('steps', 25)
    rescale_t = params.get('rescale_t', 1.0)
    cfg_strength = params.get('cfg_strength', 7.5)
    cfg_interval = params.get('cfg_interval', (0.0, 1.0))
    neg_cond = cond.get('neg_cond')

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

    args = {
        'neg_cond': neg_cond,
        'cfg_strength': cfg_strength,
        'cfg_interval': cfg_interval,
    }

    start_step = int(project_after_frac * steps)
    label = (f"P-noise init, rescale={rescale_noise}, alpha={blend_alpha}, "
             f"every={project_every}, start@step{start_step}/{steps}")
    print(f"Running Approach 2: {steps} steps, {label}")

    # --- Denoising loop ---
    for step_idx, (t, t_prev) in enumerate(t_pairs):
        # Denoise: run pretrained TRELLIS transformer
        out = sampler.sample_once(flow_model, sample, t, t_prev, cond['cond'], **args)
        sample = out.pred_x_prev

        # Project back to P superquadric features and broadcast
        should_project = (
            step_idx >= start_step and
            ((step_idx - start_step) % project_every == 0)
        )
        if should_project and blend_alpha > 0:
            z_bar, _ = project_onto_sq_subspace(sample.feats, W)
            if blend_alpha >= 1.0:
                blended = z_bar
            else:
                blended = blend_alpha * z_bar + (1.0 - blend_alpha) * sample.feats
            sample = sample.replace(blended)

        if step_idx % 5 == 0:
            print(f"  Step {step_idx}/{steps}, t={t:.4f}")

    # Final projection
    if final_project:
        z_bar, s_star = project_onto_sq_subspace(sample.feats, W)
        sample = sample.replace(z_bar)
        print(f"  Final SQ features shape: {s_star.shape}")
    else:
        print(f"  Skipping final projection")

    # Denormalize
    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(sample.device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(sample.device)
    sample = sample * std + mean

    return sample


# ---------------------------------------------------------------------------
# Rendering and visualization helpers
# ---------------------------------------------------------------------------

def render_views(gaussian, resolution: int = 512):
    """Render 4 views of a gaussian representation."""
    from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    pitchs = [0.35] * 4
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, 10, 8)
    frames = render_utils.render_frames(
        gaussian, extrinsics, intrinsics,
        {'resolution': resolution, 'bg_color': (255, 255, 255)},
        verbose=False,
    )
    return frames['color']


def create_comparison_grid(all_views, labels, output_dir):
    """Create a comparison grid image from rendered views."""
    n_views = len(all_views[0])
    img_h, img_w = all_views[0][0].shape[:2]
    n_rows = len(labels)
    label_height = 40

    grid = np.ones((n_rows * (img_h + label_height), n_views * img_w, 3), dtype=np.uint8) * 255

    for row, (label, views) in enumerate(zip(labels, all_views)):
        y_offset = row * (img_h + label_height) + label_height
        for col, img in enumerate(views):
            x_offset = col * img_w
            grid[y_offset:y_offset + img_h, x_offset:x_offset + img_w] = img

    grid_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for row, label in enumerate(labels):
        y_pos = row * (img_h + label_height) + 10
        draw.text((10, y_pos), label, fill=(0, 0, 0), font=font)

    grid_path = os.path.join(output_dir, "comparison_grid.png")
    grid_img.save(grid_path)
    print(f"  Saved comparison grid to {grid_path}")
    return grid_path


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    sq_path: str = "gui/superquadrics/chair_sq.npz",
    text_prompt: str = "a wooden chair",
    seed: int = 42,
    tau: float = 0.02,
    t0_idx: int = 6,
    steps: int = 12,
    cfg_strength: float = 7.5,
    output_dir: str = "approach2_results",
):
    os.makedirs(output_dir, exist_ok=True)

    # --- Load SQ params ---
    print("Loading superquadric parameters...")
    sq_params = load_sq_params(sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)
    P = len(sq_params)
    print(f"  {P} superquadrics, mesh_center={mesh_center}, mesh_scale={mesh_scale:.4f}")

    # --- Build SQ mesh for spatial control ---
    import open3d as o3d
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gui'))
    from gui_text_image import add_superquadric_compact_rot_mat
    from gui.utils import merge_meshes

    meshes = []
    for sq in sq_params:
        vertices, triangles = add_superquadric_compact_rot_mat(
            sq['scale'], sq['shape'], sq['translation'], sq['rotation'], resolution=100)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        meshes.append(mesh)
    merged_mesh = merge_meshes(meshes)
    merged_mesh.translate(-mesh_center)
    merged_mesh.scale(mesh_scale, (0, 0, 0))
    spatial_control_mesh_path = os.path.join(output_dir, "spatial_control_mesh.ply")
    o3d.io.write_triangle_mesh(spatial_control_mesh_path, merged_mesh)

    # --- Load pipeline ---
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    # --- Encode conditioning ---
    cond = pipeline.get_cond_text([text_prompt])

    # --- Sample structure (shared across all configs) ---
    print("Sampling sparse structure...")
    torch.manual_seed(seed)
    spatial_control_latent = pipeline.encode_spatial_control(spatial_control_mesh_path)
    cond_with_control = {**cond, 'control': spatial_control_latent}
    coords = pipeline.sample_sparse_structure(
        cond_with_control, num_samples=1,
        sampler_params={
            "steps": steps,
            "cfg_strength": cfg_strength,
            "t0_idx_value": t0_idx,
            "spatial_control_mesh_path": spatial_control_mesh_path,
        }
    )
    print(f"  Got {coords.shape[0]} active voxels")

    # --- Compute weight matrix ---
    voxel_positions = coords_to_world_positions(coords)
    print(f"  Voxel position range: [{voxel_positions.min():.3f}, {voxel_positions.max():.3f}]")

    print("Computing superquadric weight matrix...")
    W = compute_weight_matrix(voxel_positions, sq_params, mesh_center, mesh_scale, tau=tau)
    print(f"  W shape: {W.shape}, P={P}")

    dominant = W.argmax(dim=1)
    for i in range(P):
        count = (dominant == i).sum().item()
        print(f"    SQ {i}: {count} dominant voxels")

    # --- Experiment configurations ---
    configs = [
        # Baseline: standard TRELLIS
        {"name": "Baseline (TRELLIS)",
         "method": "baseline"},

        # P-noise init only, NO projection during denoising, NO final project
        {"name": "P-noise init only (no projection)",
         "method": "approach2",
         "rescale_noise": True, "blend_alpha": 0.0,
         "project_every": 1, "project_after_frac": 0.0,
         "final_project": False},

        # P-noise init, no projection during denoising, final project only
        {"name": "P-noise init, final project only",
         "method": "approach2",
         "rescale_noise": True, "blend_alpha": 0.0,
         "project_every": 1, "project_after_frac": 0.0,
         "final_project": True},

        # Very soft blending during denoising
        {"name": "P-noise, alpha=0.05, every step",
         "method": "approach2",
         "rescale_noise": True, "blend_alpha": 0.05,
         "project_every": 1, "project_after_frac": 0.0,
         "final_project": True},

        {"name": "P-noise, alpha=0.1, every step",
         "method": "approach2",
         "rescale_noise": True, "blend_alpha": 0.1,
         "project_every": 1, "project_after_frac": 0.0,
         "final_project": True},

        # Soft blending, only in last 25% of steps
        {"name": "P-noise, alpha=0.1, last 25%",
         "method": "approach2",
         "rescale_noise": True, "blend_alpha": 0.1,
         "project_every": 1, "project_after_frac": 0.75,
         "final_project": True},
    ]

    all_views = []
    labels = []

    for cfg in configs:
        name = cfg["name"]
        method = cfg["method"]
        print(f"\n=== {name} ===")
        torch.manual_seed(seed)

        if method == "baseline":
            slat = pipeline.sample_slat(cond, coords)

        elif method == "approach1":
            slat = sample_slat_with_projection(
                pipeline, cond, coords, W,
                sampler_params={},
                project_every=cfg["project_every"],
                blend_alpha=cfg["blend_alpha"],
                project_after_frac=cfg["project_after_frac"],
                final_project=cfg["final_project"],
            )

        elif method == "approach2":
            slat = sample_slat_p_noise(
                pipeline, cond, coords, W,
                sampler_params={},
                project_every=cfg["project_every"],
                blend_alpha=cfg["blend_alpha"],
                project_after_frac=cfg["project_after_frac"],
                rescale_noise=cfg["rescale_noise"],
                final_project=cfg.get("final_project", True),
            )

        print(f"  Decoding {name}...")
        gs = pipeline.decode_slat(slat, formats=['gaussian'])['gaussian'][0]
        del slat
        torch.cuda.empty_cache()

        print(f"  Rendering {name}...")
        views = render_views(gs)
        del gs
        torch.cuda.empty_cache()

        safe_name = name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("=", "")
        for i, v in enumerate(views):
            Image.fromarray(v).save(os.path.join(output_dir, f"{safe_name}_view{i}.png"))

        all_views.append(views)
        labels.append(name)

    # --- Comparison grid ---
    print("\nCreating comparison grid...")
    grid_path = create_comparison_grid(all_views, labels, output_dir)

    # --- SQ assignment visualization ---
    print("Creating superquadric assignment visualization...")
    create_sq_assignment_viz(coords, W, sq_params, output_dir)

    print(f"\nDone! Results saved to {output_dir}/")
    return grid_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Approach 2: P-noise initialization experiment")
    parser.add_argument("--sq-path", default="gui/superquadrics/chair_sq.npz")
    parser.add_argument("--prompt", default="a wooden chair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.02,
                        help="Temperature for distance-based weighting")
    parser.add_argument("--t0-idx", type=int, default=6)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg-strength", type=float, default=7.5)
    parser.add_argument("--output-dir", default="approach2_results")
    args = parser.parse_args()

    run_experiment(
        sq_path=args.sq_path,
        text_prompt=args.prompt,
        seed=args.seed,
        tau=args.tau,
        t0_idx=args.t0_idx,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        output_dir=args.output_dir,
    )

