"""
Approach 5: Local Semantic Guidance via Spatial Masks
- Strategy: Divide Superquadrics into semantic groups (Top, Mid, Bottom).
- Evaluate the Flow Model separately for each local prompt.
- Spatially fuse the velocity/predictions using the SQ distance masks.
- Solves: Attribute Bleeding (Color entanglement) present in global text conditioning.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils

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
    f = (torch.abs(x/ax)**(2/e2) + torch.abs(y/ay)**(2/e2))**(e2/e1) + torch.abs(z/az)**(2/e1)
    f = f.clamp(min=1e-12)
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
# 2. Semantic Grouping (Auto-detect Backrest, Seat, Legs by Height)
# ---------------------------------------------------------------------------

def group_sqs_by_height(sq_params, mesh_center):
    """Automatically group superquadrics into Bottom (0), Middle (1), Top (2) based on vertical axis."""
    centers = np.array([sq['translation'] for sq in sq_params])
    # Find the vertical axis (axis with largest variance)
    vertical_axis = np.argmax(np.var(centers, axis=0))
    z_coords = centers[:, vertical_axis]
    
    # Simple 1D clustering using percentiles
    z_min, z_max = z_coords.min(), z_coords.max()
    h = z_max - z_min
    
    group_map = {}
    for i, z in enumerate(z_coords):
        if z < z_min + 0.35 * h:
            group_map[i] = 0  # Bottom (Legs)
        elif z > z_max - 0.35 * h:
            group_map[i] = 2  # Top (Backrest)
        else:
            group_map[i] = 1  # Middle (Seat)
            
    print(f"Auto-Semantic Grouping (Axis {vertical_axis}):")
    print(f"  Bottom (Legs) : {[k for k,v in group_map.items() if v==0]}")
    print(f"  Middle (Seat) : {[k for k,v in group_map.items() if v==1]}")
    print(f"  Top (Backrest): {[k for k,v in group_map.items() if v==2]}")
    return group_map

# ---------------------------------------------------------------------------
# 3. Compositional Sampling (The Magic of Exp 5)
# ---------------------------------------------------------------------------

@torch.no_grad() 
def sample_slat_compositional(
    pipeline, coords, W, group_map, conds_dict, 
    steps=12, cfg_strength=7.5
):
    flow_model = pipeline.models['slat_flow_model_text']
    N, D = coords.shape[0], flow_model.in_channels
    device = pipeline.device

    W_semantic = torch.zeros((N, 3), device=device)
    for sq_idx, group_idx in group_map.items():
        W_semantic[:, group_idx] += W[:, sq_idx]
    
    z_init = torch.randn(N, D, device=device)
    sample = sp.SparseTensor(feats=z_init, coords=coords)
    sampler = pipeline.slat_sampler
    t_seq = np.linspace(1, 0, steps + 1)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
    
    print("Sampling with Local Semantic Guidance (Compositional Fusion)...")

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        dt = t_prev - t
        feats_fused = torch.zeros_like(sample.feats)

        for group_idx, cond in conds_dict.items():
            out = sampler.sample_once(
                flow_model, sample, t, t_prev, cond['cond'], 
                cfg_strength=cfg_strength, neg_cond=cond.get('neg_cond'), cfg_interval=(0.0, 1.0)
            )
            mask = W_semantic[:, group_idx:group_idx+1]
            feats_fused += mask * out.pred_x_prev.feats
            
            del out 
            
        v_fused = (feats_fused - sample.feats) / dt
        sample = sample.replace(sample.feats + dt * v_fused)
        
        if step_idx % 3 == 0: print(f"    Step {step_idx}/{steps}")

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)
    return sample * std + mean

# ---------------------------------------------------------------------------
# 4. Rendering and Visualization with Text
# ---------------------------------------------------------------------------

def create_labeled_grid(baseline_imgs, exp5_imgs, global_prompt, local_prompts, output_path):
    img_h, img_w, _ = baseline_imgs[0].shape
    header_h = 100
    row_h = img_h + header_h
    n_cols = len(baseline_imgs)
    
    canvas = Image.new('RGB', (n_cols * img_w, row_h * 2), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a nice font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Draw Baseline
    draw.text((10, 10), "ROW 1: Baseline (Standard TRELLIS)", fill="black", font=font)
    draw.text((10, 40), f"Global Prompt: '{global_prompt}'", fill="red", font=font_small)
    draw.text((10, 65), "Result: Attribute Bleeding (Fails to separate colors)", fill="gray", font=font_small)
    
    for i, img in enumerate(baseline_imgs):
        canvas.paste(Image.fromarray(img), (i * img_w, header_h))

    # Draw Exp 5
    y_offset = row_h
    draw.text((10, y_offset + 10), "ROW 2: Experiment 5 (Local Semantic Guidance via Spatial Masks)", fill="black", font=font)
    draw.text((10, y_offset + 40), f"Top SQ: '{local_prompts[2]}' | Mid SQ: '{local_prompts[1]}'", fill="blue", font=font_small)
    draw.text((10, y_offset + 65), f"Bottom SQ: '{local_prompts[0]}'", fill="blue", font=font_small)
    
    for i, img in enumerate(exp5_imgs):
        canvas.paste(Image.fromarray(img), (i * img_w, y_offset + header_h))

    canvas.save(output_path)
    print(f"\nAwesome! Comparison grid saved to: {output_path}")

# ---------------------------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------------------------

def run_experiment():
    output_dir = "approach5_results"
    os.makedirs(output_dir, exist_ok=True)
    
    sq_path = "gui/superquadrics/chair_sq.npz"
    steps = 15
    seed = 42

    global_prompt = "a plastic chair, strictly solid red seat, solid blue backrest, and solid yellow legs"
    local_prompts_text = {
        0: "a plastic chair with solid yellow legs",
        1: "a plastic chair with a solid red seat",
        2: "a plastic chair with a solid blue backrest"
    }

    print("1. Loading Geometry...")
    sq_params = load_sq_params(sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)
    group_map = group_sqs_by_height(sq_params, mesh_center)

    print("2. Loading Pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda() 

    cond_global = pipeline.get_cond_text([global_prompt])
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts_text.items()}
    
    print("3. Sampling Base Structure...")
    torch.manual_seed(seed)
    coords = pipeline.sample_sparse_structure(cond_global, num_samples=1, sampler_params={"steps": steps})
    W = compute_hard_W(coords_to_world(coords), sq_params, mesh_center, mesh_scale)

    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0, np.pi/2, np.pi, 3*np.pi/2], [0.35]*4, 10, 8)

    # ================== RUN BASELINE ==================
    print("\n--- Running Baseline ---")
    torch.manual_seed(seed)
    slat_baseline = pipeline.sample_slat(cond_global, coords, sampler_params={"steps": steps})
    gs_baseline = pipeline.decode_slat(slat_baseline, formats=['gaussian'])['gaussian'][0]
    
    print("Rendering Baseline...")
    frames_baseline = render_utils.render_frames(gs_baseline, extr, intr, {'resolution': 512, 'bg_color': (255,255,255)})['color']

    del slat_baseline
    del gs_baseline
    torch.cuda.empty_cache()
    print("Baseline cleared from VRAM.")

    # ================== RUN EXP 5 ==================
    print("\n--- Running Experiment 5 ---")
    torch.manual_seed(seed)
    slat_exp5 = sample_slat_compositional(pipeline, coords, W, group_map, conds_local, steps=steps)
    gs_exp5 = pipeline.decode_slat(slat_exp5, formats=['gaussian'])['gaussian'][0]
    
    print("Rendering Exp 5...")
    frames_exp5 = render_utils.render_frames(gs_exp5, extr, intr, {'resolution': 512, 'bg_color': (255,255,255)})['color']

    del slat_exp5
    del gs_exp5
    torch.cuda.empty_cache()

    # ================== CREATE GRID ==================
    print("\nSaving comparison grid...")
    out_path = os.path.join(output_dir, "exp5_vs_baseline_color_routing.png")
    create_labeled_grid(frames_baseline, frames_exp5, global_prompt, local_prompts_text, out_path)

if __name__ == "__main__":
    run_experiment()