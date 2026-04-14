"""
Approach 6: Extreme Composition (Per-Superquadric Semantic Routing)
- Strategy: Map every single Superquadric to a unique text condition.
- Stress test for localized geometry/material generation.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)
os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils
from approach1_experiment import coords_to_world, compute_mesh_normalization, load_sq_params

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

@torch.no_grad()
def sample_slat_extreme(pipeline, coords, W, conds_dict, steps=15, cfg_strength=7.5):
    """
    conds_dict: { sq_index (int) : cond (dict) }
    W: (N, P) where P is the number of SQs
    """
    flow_model = pipeline.models['slat_flow_model_text']
    N, D = coords.shape[0], flow_model.in_channels
    device = pipeline.device

    z_init = torch.randn(N, D, device=device)
    sample = sp.SparseTensor(feats=z_init, coords=coords)
    sampler = pipeline.slat_sampler
    t_seq = np.linspace(1, 0, steps + 1)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
    
    P = W.shape[1]
    print(f"Sampling Extreme Composition: Integrating {P} separate prompts per step...")

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        dt = t_prev - t
        feats_fused = torch.zeros_like(sample.feats)
        
        for sq_idx, cond in conds_dict.items():
            out = sampler.sample_once(
                flow_model, sample, t, t_prev, cond['cond'], 
                cfg_strength=cfg_strength, neg_cond=cond.get('neg_cond'), cfg_interval=(0.0, 1.0)
            )
            mask = W[:, sq_idx:sq_idx+1]
            feats_fused += mask * out.pred_x_prev.feats
            del out 
            
        v_fused = (feats_fused - sample.feats) / dt
        sample = sample.replace(sample.feats + dt * v_fused)
        if step_idx % 3 == 0: print(f"    Step {step_idx}/{steps}")

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)
    return sample * std + mean

def run_experiment():
    output_dir = "approach6_results"
    os.makedirs(output_dir, exist_ok=True)
    
    sq_path = "gui/superquadrics/chair_sq.npz"
    steps = 15
    seed = 42

    print("1. Loading Geometry...")
    sq_params = load_sq_params(sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)
    P = len(sq_params)
    print(f"Found {P} Superquadrics in file.")

    local_prompts_text = {
        0: "a pink plastic chair leg",       
        5: "a blue plastic chair leg",    
        1: "a yellow plastic chair leg",                   
        3: "a chair leg made of natural brown wood",                 

        7: "a green plastic chair backrest",  
        
        8: "a chair seat cushion made of soft red velvet fabric",   
        
        2: "a silver chair crossbar",   
        4: "a silver chair crossbar",  
        6: "a silver chair crossbar",  
        9: "a silver chair crossbar",  
    }
    
    if len(local_prompts_text) != P:
        print(f"Warning: You defined {len(local_prompts_text)} prompts but there are {P} SQs.")
        for i in range(P):
            if i not in local_prompts_text:
                local_prompts_text[i] = "a plain gray plastic chair part"

    print("2. Loading Pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda() 

    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts_text.items()}
    
    print("3. Sampling Base Structure...")
  
    global_structure_prompt = "a minimalist chair with four thin legs, crossbars, a seat cushion, and a backrest"
    cond_struct = pipeline.get_cond_text([global_structure_prompt])
    
    torch.manual_seed(seed)
    coords = pipeline.sample_sparse_structure(cond_struct, num_samples=1, sampler_params={"steps": steps})
    W = compute_hard_W(coords_to_world(coords), sq_params, mesh_center, mesh_scale)

    # ================== RUN EXTREME EXP ==================
    print("\n--- Running Experiment 6 (Extreme) ---")
    torch.manual_seed(seed)
    slat_exp6 = sample_slat_extreme(pipeline, coords, W, conds_local, steps=steps)
    gs_exp6 = pipeline.decode_slat(slat_exp6, formats=['gaussian'])['gaussian'][0]
    
    print("Rendering...")
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0, np.pi/2, np.pi, 3*np.pi/2], [0.35]*4, 10, 8)
    frames_exp6 = render_utils.render_frames(gs_exp6, extr, intr, {'resolution': 512, 'bg_color': (255,255,255)})['color']
    

    row_img = np.concatenate(frames_exp6, axis=1)
    Image.fromarray(row_img).save(os.path.join(output_dir, "extreme_6_materials_chair.png"))
    print(f"\nDone! Extreme result saved to: {output_dir}/extreme_6_materials_chair.png")
    

    with open(os.path.join(output_dir, "extreme_prompts.txt"), "w") as f:
        for k, v in local_prompts_text.items():
            f.write(f"SQ {k}: {v}\n")

if __name__ == "__main__":
    run_experiment()