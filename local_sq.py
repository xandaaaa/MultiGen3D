import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
experiments_dir = os.path.join(project_root, "experiments")
sys.path.insert(0, project_root)
sys.path.insert(0, experiments_dir)
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

def compute_soft_W(voxel_pos, sq_params, mesh_center, mesh_scale, tau=0.02):
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
    return torch.softmax(-dist / tau, dim=1)

def make_contextual_local_prompts(global_prompt, local_prompts):
    """Keep object identity in each per-SQ condition while emphasizing the local part."""
    return {
        sq_idx: f"{global_prompt}. This object part is {local_prompt}."
        for sq_idx, local_prompt in local_prompts.items()
    }

@torch.no_grad()
def sample_slat_regional_refine(pipeline, coords, W, conds_local, cond_global,
                                 global_steps=25, refine_steps=10, t_noise=0.5,
                                 cfg_strength=7.5, rescale_t=3.0, local_blend=0.9,
                                 mask_threshold=0.02):
    """
    Two-stage approach:
      1. Generate a coherent global SLAT with the global prompt (stays on-manifold).
      2. For each SQ region, refine with the local prompt starting from the global
         SLAT partially noised at level t_noise (inpainting style).
         Non-masked voxels are reprojected at the correct noise level each step
         so the model always has globally-coherent context during local recoloring.

    This avoids the per-step masking corruption of sample_slat_extreme, where
    mixing 10 velocity fields every step drives the shared state off-manifold and
    causes feature collapse to gray by step 25.
    """
    device = pipeline.device
    flow_model = pipeline.models['slat_flow_model_text']
    sampler = pipeline.slat_sampler
    null_cond = pipeline.text_cond_model['null_cond']

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)

    # --- Stage 1: global generation ---
    print("Stage 1: global generation...")
    noise_global = torch.randn(coords.shape[0], flow_model.in_channels, device=device)
    sample_g = sp.SparseTensor(feats=noise_global, coords=coords)
    t_seq = np.linspace(1, 0, global_steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
        out = sampler.sample_once(flow_model, sample_g, t, t_prev, cond_global['cond'],
                                  cfg_strength=cfg_strength, neg_cond=cond_global.get('neg_cond'),
                                  cfg_interval=(0.5, 0.95))
        sample_g = sample_g.replace(out.pred_x_prev.feats)
        del out

    # Normalized global clean features (in flow-model latent space, before mean/std)
    x0_global = sample_g.feats  # (N, D), this is x_0 in normalized space

    # --- Stage 2: per-SQ regional refinement ---
    result_feats = x0_global.clone()

    t_seq_r_raw = np.linspace(t_noise, 0, refine_steps + 1)
    t_seq_r = rescale_t * t_seq_r_raw / (1 + (rescale_t - 1) * t_seq_r_raw)
    t_pairs_r = list((t_seq_r[i], t_seq_r[i + 1]) for i in range(refine_steps))
    t_start = float(t_seq_r[0])

    P = W.shape[1]
    print(
        f"Stage 2: refining {P} SQ regions "
        f"(t_noise={t_noise}, {refine_steps} steps each, local_blend={local_blend})..."
    )

    for sq_idx, cond_local in conds_local.items():
        mask_weight = W[:, sq_idx:sq_idx + 1].clamp(0.0, 1.0)
        active = mask_weight > mask_threshold
        if not active.any():
            continue

        # Fixed noise for this region — used to reprojected non-masked voxels each step
        noise_fixed = torch.randn_like(x0_global)

        # Initial state: global clean + noise_fixed blended at t_noise for ALL voxels
        feats_init = (1 - t_start) * x0_global + t_start * noise_fixed
        sample_r = sp.SparseTensor(feats=feats_init, coords=coords)

        for t, t_prev in t_pairs_r:
            out = sampler.sample_once(
                flow_model, sample_r, t, t_prev, cond_local['cond'],
                cfg_strength=cfg_strength, neg_cond=null_cond,
                cfg_interval=(0.0, min(0.95, t_start + 0.05)),
            )
            # Masked region: take local branch's denoised features
            # Non-masked region: reproject global clean to correct noise level t_prev
            #   x_{t_prev} = (1 - t_prev)*x0_global + t_prev*noise_fixed
            feats_nonmask = (1 - t_prev) * x0_global + t_prev * noise_fixed
            new_feats = feats_nonmask + mask_weight * (out.pred_x_prev.feats - feats_nonmask)
            sample_r = sample_r.replace(new_feats)
            del out

        result_feats = result_feats + local_blend * mask_weight * (sample_r.feats - x0_global)
        print(f"    SQ {sq_idx} done")

    # Apply pipeline normalization (mean/std) for decoder
    sample_out = sp.SparseTensor(feats=result_feats * std + mean, coords=coords)
    return sample_out


# Keep old extreme sampler for reference / ablation
@torch.no_grad()
def sample_slat_extreme(pipeline, coords, W, conds_dict, cond_global,
                        steps=25, cfg_strength=7.5, rescale_t=3.0,
                        detail_t_threshold=0.5):
    flow_model = pipeline.models['slat_flow_model_text']
    N, D = coords.shape[0], flow_model.in_channels
    device = pipeline.device
    null_cond = pipeline.text_cond_model['null_cond']

    z_init = torch.randn(N, D, device=device)
    sample = sp.SparseTensor(feats=z_init, coords=coords)
    sampler = pipeline.slat_sampler
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

    P = W.shape[1]
    print(f"Sampling Extreme Composition: {P} prompts, detail threshold t={detail_t_threshold}")

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        feats_fused = torch.zeros_like(sample.feats)
        neg = cond_global['cond'] if t >= detail_t_threshold else null_cond

        for sq_idx, cond in conds_dict.items():
            out = sampler.sample_once(
                flow_model, sample, t, t_prev, cond['cond'],
                cfg_strength=cfg_strength, neg_cond=neg,
                cfg_interval=(0.0, 0.95),
            )
            mask = W[:, sq_idx:sq_idx + 1]
            feats_fused += mask * out.pred_x_prev.feats
            del out

        sample = sample.replace(feats_fused)
        if step_idx % 5 == 0:
            print(f"    Step {step_idx}/{steps}  neg={'global' if t >= detail_t_threshold else 'null'}")

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)
    return sample * std + mean

def run_experiment():
    output_dir = os.path.join(project_root, "approach6_results")
    os.makedirs(output_dir, exist_ok=True)
    
    sq_path = os.path.join(project_root, "gui", "superquadrics", "chair_sq.npz")
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
    pipeline = TrellisTextTo3DPipeline.from_pretrained(os.path.join(project_root, "gui"))
    pipeline.cuda() 

    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts_text.items()}

    print("3. Sampling Base Structure...")

    global_structure_prompt = "a minimalist chair with four thin legs, crossbars, a seat cushion, and a backrest"
    cond_global = pipeline.get_cond_text([global_structure_prompt])

    torch.manual_seed(seed)
    coords = pipeline.sample_sparse_structure(cond_global, num_samples=1, sampler_params={"steps": steps})
    W = compute_hard_W(coords_to_world(coords), sq_params, mesh_center, mesh_scale)

    # ================== RUN EXTREME EXP ==================
    print("\n--- Running Experiment 6 (Extreme) ---")
    torch.manual_seed(seed)
    slat_exp6 = sample_slat_regional_refine(pipeline, coords, W, conds_local, cond_global,
                                             global_steps=steps)
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
