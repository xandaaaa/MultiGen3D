import os
import sys
import torch
import numpy as np
from PIL import Image
import argparse
import gc

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)
os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils
from approach1_experiment import coords_to_world, compute_mesh_normalization, load_sq_params
from approach5_experiment import superquadric_radial_distance

# --- 1. Soft Assignment ---
def compute_soft_W(voxel_pos, sq_params, mesh_center, mesh_scale, temperature=0.01):
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

    if temperature <= 0:
        # Hard Assignment 
        W = torch.zeros_like(dist)
        # Find indices of the minimum distance for each voxel
        min_indices = torch.argmin(dist, dim=1, keepdim=True)
        # Set those to 1
        W.scatter_(1, min_indices, 1.0)
    else:
        # Soft Assignment (Softmax)
        W = torch.softmax(-dist / (temperature + 1e-6), dim=1)    

    return W

# --- 2. Advanced Sampling Loop ---
@torch.no_grad()
def sample_slat_advanced(pipeline, coords, W, cond_global, conds_local, alpha, threshold, steps=15, cfg_strength=7.5):
    flow_model = pipeline.models['slat_flow_model_text']
    sampler = pipeline.slat_sampler
    device = pipeline.device
    N, D = coords.shape[0], flow_model.in_channels

    z_init = torch.randn(N, D, device=device)
    sample = sp.SparseTensor(feats=z_init, coords=coords)
    t_seq = np.linspace(1, 0, steps + 1)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

    print(f"Sampling with alpha={alpha}, threshold={threshold}...")

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        dt = t_prev - t
        
        # 1. Global Velocity
        out_global = sampler.sample_once(flow_model, sample, t, t_prev, cond_global['cond'], cfg_strength=cfg_strength, neg_cond=cond_global.get('neg_cond'), cfg_interval=(0.0, 1.0))
        v_global = (out_global.pred_x_prev.feats - sample.feats) / dt
        
        # 2. Delayed Intervention Logic
        if t > threshold:
            v_final = v_global # Only use global structure guidance
        else:
            # Calculate Fused Velocity (Local Guidance)
            feats_fused = torch.zeros_like(sample.feats)
            for sq_idx, cond in conds_local.items():
                out_local = sampler.sample_once(flow_model, sample, t, t_prev, cond['cond'], cfg_strength=cfg_strength, neg_cond=cond_global.get('neg_cond'), cfg_interval=(0.0, 1.0))
                mask = W[:, sq_idx:sq_idx+1]
                feats_fused += mask * out_local.pred_x_prev.feats
                del out_local
            
            v_fused = (feats_fused - sample.feats) / dt
            # Linear Interpolation
            v_final = alpha * v_global + (1.0 - alpha) * v_fused
        
        sample = sample.replace(sample.feats + dt * v_final)
        del out_global
        if step_idx % 3 == 0: print(f"    Step {step_idx}/{steps} | t={t:.2f}")

    # Denormalize
    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)
    return sample * std + mean

# --- 3. Run ---
def run_experiment(alpha, threshold, temp):
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Start: alpha={alpha}, thresh={threshold}, temp={temp}")

    output_dir = "/work/courses/3dv/team4/MultiGen3D/exp_6_outputs/seed1024"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Experiment results will be saved to: {output_dir}")
    
    # Init Pipeline
    sq_path = "gui/superquadrics/chair_sq.npz"
    sq_params = load_sq_params(sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda() 
    print(pipeline.sparse_structure_sampler_params)
    
    global_prompt = "a Bauhaus chair"

    local_prompts = { i: "a Bauhaus chair" for i in range(len(sq_params)) }
    local_prompts.update({0: "a yellow chair leg", 1: "a yellow chair leg", 3: "a yellow chair leg", 5: "a yellow chair leg", 7: "a green backrest", 8: "a red seat cushion"})

    cond_global = pipeline.get_cond_text([global_prompt])
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts.items()}

    seed = 1024
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Structure
    coords = pipeline.sample_sparse_structure(cond_global, num_samples=1)
    W = compute_soft_W(coords_to_world(coords), sq_params, mesh_center, mesh_scale, temperature=temp)
    
    # Sample
    slat_res = sample_slat_advanced(pipeline, coords, W, cond_global, conds_local, alpha, threshold)
    del W, coords 
    gc.collect()
    torch.cuda.empty_cache()

    filename = f"chair_alpha{alpha}_th{threshold}_temp{temp}.png"
    save_path = os.path.join(output_dir, filename)
    
    # Render
    gs_res = pipeline.decode_slat(slat_res, formats=['gaussian'])['gaussian'][0]
    del slat_res
    gc.collect()
    torch.cuda.empty_cache()

    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0, np.pi/2, np.pi, 3*np.pi/2], [0.35]*4, 10, 8)
    frames = render_utils.render_frames(gs_res, extr, intr, {'resolution': 512, 'bg_color': (255,255,255)})['color']
    
    Image.fromarray(np.concatenate(frames, axis=1)).save(save_path)
    del gs_res, frames
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=0.01)
    args = parser.parse_args()
    run_experiment(args.alpha, args.threshold, args.temp)