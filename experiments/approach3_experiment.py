"""
Approach 3: Optimized Velocity Consistency Experiment
- Scheme A: Blend Alpha (Mixing consistent velocity with original model flow)
- Scheme B: Project After Fraction (Start constraint only in later stages)
- Scheme C: Rescale Noise (Ensures initial latent variance is 1.0)
- Baseline: Original TRELLIS sampling included for direct comparison
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --- Environment Setup: Fix for ModuleNotFoundError ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

# Re-using validated utilities
from approach1_experiment import (
    coords_to_world_positions,
    compute_mesh_normalization,
    load_sq_params,
    create_sq_assignment_viz,
)

# ---------------------------------------------------------------------------
# 1. Geometry: Radial Distance Calculation
# ---------------------------------------------------------------------------

def superquadric_radial_distance(x_local, semi_axes, eps):
    e1, e2 = eps[0].clamp(min=0.01), eps[1].clamp(min=0.01)
    ax, ay, az = semi_axes[0].clamp(min=1e-6), semi_axes[1].clamp(min=1e-6), semi_axes[2].clamp(min=1e-6)
    x, y, z = x_local[:, 0], x_local[:, 1], x_local[:, 2]
    # f(x) calculation
    f = (torch.abs(x/ax)**(2/e2) + torch.abs(y/ay)**(2/e2))**(e2/e1) + torch.abs(z/az)**(2/e1)
    f = f.clamp(min=1e-12)
    # Radial distance d_r
    return torch.norm(x_local, dim=-1) * torch.abs(1.0 - f**(-e1/2.0))

def compute_hard_W(voxel_pos, sq_params, mesh_center, mesh_scale):
    device = voxel_pos.device
    N, P = voxel_pos.shape[0], len(sq_params)
    dist = torch.zeros(N, P, device=device)
    
    # Pre-convert normalization params to float tensor
    m_center = torch.tensor(mesh_center, device=device).float()
    
    for i, sq in enumerate(sq_params):
        # Convert all numpy inputs to float32 tensors
        c = (torch.tensor(sq['translation'], device=device).float() - m_center) * mesh_scale
        rot = torch.tensor(sq['rotation'], device=device).float()
        s = torch.tensor(sq['scale'], device=device).float() * mesh_scale
        e = torch.tensor(sq['shape'], device=device).float()
        
        # Matrix multiplication now has matching dtypes (float32)
        x_loc = (voxel_pos - c.unsqueeze(0)) @ rot 
        dist[:, i] = superquadric_radial_distance(x_loc, s, e)
        
    W = torch.zeros((N, P), device=device)
    # Hard assignment with tie-breaking
    W.scatter_(1, (dist + torch.randn_like(dist)*1e-8).argmin(1).unsqueeze(1), 1.0)
    return W    

# ---------------------------------------------------------------------------
# 2. Optimized Sampling: Velocity Consistency with Schemes A, B, C
# ---------------------------------------------------------------------------

def sample_slat_optimized(
    pipeline, cond, coords, W, 
    steps=12, 
    cfg_strength=7.5,
    blend_alpha=1.0,        # Scheme A: Blend factor [0, 1]
    project_after_frac=0.0, # Scheme B: Fraction of steps to start constraints
    rescale_noise=True      # Scheme C: Rescale initial latent variance
):
    flow_model = pipeline.models['slat_flow_model_text'] if cond['cond'].shape[-1] == 768 else pipeline.models['slat_flow_model_image']
    N, P = W.shape
    D = flow_model.in_channels
    device = pipeline.device

    # --- Scheme C: Rescale Noise ---
    s_init = torch.randn(P, D, device=device)
    z_init = W @ s_init
    if rescale_noise:
        # Normalize to unit variance (avoids high-saturation 'neon' artifacts)
        z_init = z_init / (z_init.std() + 1e-8)
    
    sample = sp.SparseTensor(feats=z_init, coords=coords)
    
    t_seq = np.linspace(1, 0, steps + 1)
    rescale_t = pipeline.slat_sampler_params.get('rescale_t', 1.0)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
    
    start_step = int(project_after_frac * steps)

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        dt = t_prev - t
        out = pipeline.slat_sampler.sample_once(
            flow_model, sample, t, t_prev, cond['cond'], 
            cfg_strength=cfg_strength, neg_cond=cond.get('neg_cond'),
            cfg_interval=(0.0, 1.0)
        )
        
        # Calculate the model's suggested velocity
        v_original = (out.pred_x_prev.feats - sample.feats) / dt
        
        # --- Scheme A & B: Consistency Constraints ---
        if step_idx >= start_step:
            # Average velocity within the primitive
            counts = W.sum(0).unsqueeze(1).clamp(min=1)
            v_consistent = W @ ((W.T @ v_original) / counts)
            # Mix consistent velocity with original flow
            v_final = blend_alpha * v_consistent + (1.0 - blend_alpha) * v_original
        else:
            v_final = v_original

        sample = sample.replace(sample.feats + dt * v_final)
        if step_idx % 5 == 0: print(f"    Step {step_idx}/{steps}")

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)
    return sample * std + mean

# ---------------------------------------------------------------------------
# 3. Comparison Experiment Loop
# ---------------------------------------------------------------------------

def run_experiment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sq_params = load_sq_params(args.sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)

    print("Loading pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    cond = pipeline.get_cond_text([args.prompt])
    
    # Pre-generate Structure
    print("Generating Structure...")
    torch.manual_seed(args.seed)
    coords = pipeline.sample_sparse_structure(cond, num_samples=1, sampler_params={"steps": args.steps})
    voxel_pos = coords_to_world_positions(coords)
    W = compute_hard_W(voxel_pos, sq_params, mesh_center, mesh_scale)

    # Experiment Table
    configs = [
        {"name": "baseline",    "alpha": 0.0, "frac": 0.0, "rescale": False},
        {"name": "hard_consist", "alpha": 1.0, "frac": 0.0, "rescale": False}, # Your previous result
        {"name": "blend_alpha_03", "alpha": 0.3, "frac": 0.0, "rescale": True}, # Recommended soft guide
        {"name": "late_consist", "alpha": 1.0, "frac": 0.5, "rescale": True},  # Guide in second half
    ]

    all_views = []
    
    for cfg in configs:
        print(f"\n--- Running: {cfg['name']} ---")
        torch.manual_seed(args.seed)
        
        if cfg['name'] == "baseline":
            slat = pipeline.sample_slat(cond, coords, sampler_params={"steps": args.steps})
        else:
            slat = sample_slat_optimized(
                pipeline, cond, coords, W, 
                steps=args.steps, 
                blend_alpha=cfg['alpha'], 
                project_after_frac=cfg['frac'],
                rescale_noise=cfg['rescale']
            )

        gs = pipeline.decode_slat(slat, formats=['gaussian'])['gaussian'][0]
        
        # Render
        extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0, np.pi/2, np.pi, 3*np.pi/2], [0.35]*4, 10, 8)
        frames = render_utils.render_frames(gs, extr, intr, {'resolution': 512, 'bg_color': (255,255,255)})
        
        row_img = np.concatenate(frames['color'], axis=1)
        # File name using manual experiment name prefix
        save_path = os.path.join(args.output_dir, f"{args.exp_name}_{cfg['name']}.png")
        Image.fromarray(row_img).save(save_path)
        print(f"  Saved to {save_path}")
        all_views.append(row_img)

    # Save final grid
    grid_img = np.concatenate(all_views, axis=0)
    grid_save_path = os.path.join(args.output_dir, f"{args.exp_name}_comparison_grid.png")
    Image.fromarray(grid_img).save(grid_save_path)
    print(f"\nExperiment Complete. Combined grid saved to: {grid_save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sq-path", default="gui/superquadrics/chair_sq.npz")
    parser.add_argument("--prompt", default="a wooden chair")
    parser.add_argument("--exp-name", required=True, help="Prefix for generated filenames")
    parser.add_argument("--output-dir", default="approach3_results")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args)