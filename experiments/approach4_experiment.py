"""
Approach 4: Refined Sampling with Soft Residual Guidance
- Strategy: Use Independent voxel noise (preserves shape) + Soft Velocity Guidance (aligns color).
- Fixes: Geometry collapse, Neon color artifacts, and NoneType Pipeline error.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple

# --- CRITICAL FIX FOR PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

# Reusing validated utilities from approach1_experiment.py
from approach1_experiment import (
    coords_to_world,
    compute_mesh_normalization,
    load_sq_params,
)

# ---------------------------------------------------------------------------
# 1. Geometry: Radial Distance Logic
# ---------------------------------------------------------------------------

def superquadric_radial_distance(x_local, semi_axes, eps):
    e1, e2 = eps[0].clamp(min=0.01), eps[1].clamp(min=0.01)
    ax, ay, az = semi_axes[0].clamp(min=1e-6), semi_axes[1].clamp(min=1e-6), semi_axes[2].clamp(min=1e-6)
    x, y, z = x_local[:, 0], x_local[:, 1], x_local[:, 2]
    # f(x)
    f = (torch.abs(x/ax)**(2/e2) + torch.abs(y/ay)**(2/e2))**(e2/e1) + torch.abs(z/az)**(2/e1)
    f = f.clamp(min=1e-12)
    # d_r
    return torch.norm(x_local, dim=-1) * torch.abs(1.0 - f**(-e1/2.0))

def compute_hard_W(voxel_pos, sq_params, mesh_center, mesh_scale):
    device = voxel_pos.device
    N, P = voxel_pos.shape[0], len(sq_params)
    dist = torch.zeros(N, P, device=device)
    m_center = torch.tensor(mesh_center, device=device).float()
    
    for i, sq in enumerate(sq_params):
        c = (torch.tensor(sq['translation'], device=device).float() - m_center) * mesh_scale
        rot = torch.tensor(sq['rotation'], device=device).float()
        s = torch.tensor(sq['scale'], device=device).float() * mesh_scale
        e = torch.tensor(sq['shape'], device=device).float()
        x_loc = (voxel_pos - c.unsqueeze(0)) @ rot 
        dist[:, i] = superquadric_radial_distance(x_loc, s, e)
        
    W = torch.zeros((N, P), device=device)
    W.scatter_(1, (dist + torch.randn_like(dist)*1e-8).argmin(1).unsqueeze(1), 1.0)
    return W

# ---------------------------------------------------------------------------
# 2. Denoising: Soft Residual Guidance
# ---------------------------------------------------------------------------

def sample_slat_refined(
    pipeline, cond, coords, W, 
    steps=12, 
    cfg_strength=7.5,
    guidance_strength=0.15, 
    start_frac=0.2          
):
    flow_model = pipeline.models['slat_flow_model_text'] if cond['cond'].shape[-1] == 768 else pipeline.models['slat_flow_model_image']
    N, D = coords.shape[0], flow_model.in_channels
    device = pipeline.device

    # Use independent noise to keep geometric details
    z_init = torch.randn(N, D, device=device)
    sample = sp.SparseTensor(feats=z_init, coords=coords)
    
    # Setup sampler params
    sampler = pipeline.slat_sampler
    params = {**pipeline.slat_sampler_params}
    rescale_t = params.get('rescale_t', 1.0)
    cfg_interval = params.get('cfg_interval', (0.0, 1.0))
    neg_cond = cond.get('neg_cond')

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
    
    start_step = int(start_frac * steps)

    print(f"Sampling with Refined Guidance: strength={guidance_strength}, start_step={start_step}")

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        dt = t_prev - t
        # Standard step
        out = sampler.sample_once(
            flow_model, sample, t, t_prev, cond['cond'], 
            cfg_strength=cfg_strength, neg_cond=neg_cond, cfg_interval=cfg_interval
        )
        
        v_model = (out.pred_x_prev.feats - sample.feats) / dt
        
        if step_idx >= start_step:
            # Average velocity
            counts = W.sum(0).unsqueeze(1).clamp(min=1)
            v_avg = W @ ((W.T @ v_model) / counts)
            # Soft Lerp to preserve per-voxel geometry while aligning color trends
            v_final = torch.lerp(v_model, v_avg, guidance_strength)
        else:
            v_final = v_model

        sample = sample.replace(sample.feats + dt * v_final)
        if step_idx % 5 == 0: print(f"    Step {step_idx}/{steps}, t={t:.4f}")

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)
    return sample * std + mean

# ---------------------------------------------------------------------------
# 3. Main Experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sq_params = load_sq_params(args.sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)

    print("Loading pipeline...")
    # --- FIXED: Separate init from .cuda() to avoid NoneType Error ---
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda() 

    print("Encoding conditioning...")
    cond = pipeline.get_cond_text([args.prompt])
    
    print("Sampling structure...")
    torch.manual_seed(args.seed)
    coords = pipeline.sample_sparse_structure(cond, num_samples=1, sampler_params={"steps": args.steps})
    
    W = compute_hard_W(coords_to_world(coords), sq_params, mesh_center, mesh_scale)

    configs = [
        {"name": "1_Baseline",    "strength": 0.0},
        {"name": "2_Refined_015", "strength": 0.15}, 
    ]

    all_views = []
    for cfg in configs:
        print(f"\n--- Running Experiment: {cfg['name']} ---")
        torch.manual_seed(args.seed)
        
        if cfg['name'] == "1_Baseline":
            slat = pipeline.sample_slat(cond, coords, sampler_params={"steps": args.steps})
        else:
            slat = sample_slat_refined(pipeline, cond, coords, W, steps=args.steps, guidance_strength=cfg['strength'])

        gs = pipeline.decode_slat(slat, formats=['gaussian'])['gaussian'][0]
        extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0, np.pi/2, np.pi, 3*np.pi/2], [0.35]*4, 10, 8)
        frames = render_utils.render_frames(gs, extr, intr, {'resolution': 512, 'bg_color': (255,255,255)})
        
        row_img = np.concatenate(frames['color'], axis=1)
        save_path = os.path.join(args.output_dir, f"{args.exp_name}_{cfg['name']}.png")
        Image.fromarray(row_img).save(save_path)
        all_views.append(row_img)

    grid_img = np.concatenate(all_views, axis=0)
    Image.fromarray(grid_img).save(os.path.join(args.output_dir, f"{args.exp_name}_grid.png"))
    print(f"\nDone. Results in {args.output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sq-path", default="gui/superquadrics/chair_sq.npz")
    parser.add_argument("--prompt", default="a wooden chair")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output-dir", default="approach4_results")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args)