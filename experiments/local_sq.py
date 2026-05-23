import os
import sys
import gc
from typing import Optional
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

_SHAPENET_TO_TRELLIS_R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)


def convert_shapenet_yup_to_trellis_zup(sq_params):
    """ShapeNet (and superdec output) is Y-up; TRELLIS renders with Z-up.
    Empirically TRELLIS-generated shapes also tend to face -X (ShapeNet nose at +X
    maps to TRELLIS rendered nose at -X), so we negate X as well.

    Known limitation: TRELLIS has no explicit orientation constraint — its
    output frame is consistent within a prompt seed but can differ across
    shapes/categories. So this fixed rotation matches most cases but a few
    (e.g. L-shaped sofas) may end up mirrored along one axis vs. the rendered
    geometry. The voxel routing is still spatially coherent within each shape,
    just potentially flipped relative to the ShapeNet part labels."""
    R = _SHAPENET_TO_TRELLIS_R
    out = []
    for sq in sq_params:
        out.append({
            'scale': sq['scale'],
            'shape': sq['shape'],
            'rotation': R @ np.asarray(sq['rotation'], dtype=np.float64),
            'translation': R @ np.asarray(sq['translation'], dtype=np.float64),
        })
    return out


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
    """Per-SQ conditioning prompts.

    Earlier version wrapped the local prompt with the global one
    (\"{global}. This object part is {local}.\") to preserve object identity,
    but that made the local embedding nearly identical to the global one, so
    refinement converged back to x0_global and had no visible effect. Using
    the bare local prompt makes the local trajectory genuinely diverge.
    """
    return {sq_idx: local_prompt for sq_idx, local_prompt in local_prompts.items()}

@torch.no_grad()
def sample_slat_regional_refine(pipeline, coords, W, conds_local, cond_global,
                                 global_steps=25, refine_steps=10, t_noise=0.5,
                                 cfg_strength=7.5, rescale_t=3.0, local_blend=0.9,
                                 mask_threshold=0.02, debug_dir: Optional[str] = None):
    # Env-var overrides for fast experimentation without re-threading kwargs
    if os.environ.get('LOCAL_SQ_T_NOISE'):
        t_noise = float(os.environ['LOCAL_SQ_T_NOISE'])
        print(f"  [env] t_noise overridden to {t_noise}")
    if os.environ.get('LOCAL_SQ_REFINE_STEPS'):
        refine_steps = int(os.environ['LOCAL_SQ_REFINE_STEPS'])
        print(f"  [env] refine_steps overridden to {refine_steps}")
    if os.environ.get('LOCAL_SQ_LOCAL_BLEND'):
        local_blend = float(os.environ['LOCAL_SQ_LOCAL_BLEND'])
        print(f"  [env] local_blend overridden to {local_blend}")
    if os.environ.get('LOCAL_SQ_CFG'):
        cfg_strength = float(os.environ['LOCAL_SQ_CFG'])
        print(f"  [env] cfg_strength overridden to {cfg_strength}")
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

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        _dbg_extr, _dbg_intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [0, np.pi / 2, np.pi, 3 * np.pi / 2], [0.35] * 4, 10, 8
        )
        _snapshot_oom = {"hit": False}

        def _snapshot(name, feats):
            if _snapshot_oom["hit"]:
                return
            snap = None
            gs = None
            frames = None
            try:
                snap = sp.SparseTensor(feats=feats * std + mean, coords=coords)
                gs = pipeline.decode_slat(snap, formats=["gaussian"])["gaussian"][0]
                scales = gs.get_scaling
                if not torch.isfinite(scales).all() or scales.max().item() > 10.0:
                    print(f"  [debug] snapshot {name}: degenerate Gaussian (max scale={scales.max().item():.3g}), skipping render")
                    return
                frames = render_utils.render_frames(
                    gs, _dbg_extr, _dbg_intr,
                    {"resolution": 512, "bg_color": (255, 255, 255)},
                )["color"]
                for j, frame in enumerate(frames):
                    Image.fromarray(frame).save(os.path.join(debug_dir, f"{name}_view_{j}.png"))
                Image.fromarray(np.concatenate(frames, axis=1)).save(
                    os.path.join(debug_dir, f"{name}_grid.png")
                )
                print(f"  [debug] snapshot {name} saved")
            except torch.cuda.OutOfMemoryError as e:
                _snapshot_oom["hit"] = True
                print(f"  [debug] snapshot {name}: OOM ({e}); skipping remaining snapshots this prompt")
            except Exception as e:
                print(f"  [debug] snapshot {name}: failed ({type(e).__name__}: {e})")
            finally:
                del snap, gs, frames
                gc.collect()
                torch.cuda.empty_cache()

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

    if debug_dir is not None:
        _snapshot("00_x0_global", x0_global)

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
        mask_bool = (W[:, sq_idx:sq_idx + 1] > 0.3)
        active = mask_weight > mask_threshold
        if not active.any():
            continue

        # Fixed noise for this region — used to reprojected non-masked voxels each step
        noise_fixed = torch.randn_like(x0_global)

        # Initial state: matches approach6 — blend at RAW t_noise (not rescaled t_start).
        # The schedule below uses rescaled timesteps; feeding the model an input that's
        # less noisy than its timestep claims appears to give a stronger localization
        # signal in practice (this is what approach6_experiment.py does and it works).
        feats_init = (1 - t_noise) * x0_global + t_noise * noise_fixed
        sample_r = sp.SparseTensor(feats=feats_init, coords=coords)

        for t, t_prev in t_pairs_r:
            out = sampler.sample_once(
                flow_model, sample_r, t, t_prev, cond_local['cond'],
                cfg_strength=cfg_strength, neg_cond=null_cond,
                cfg_interval=(0.0, t_noise + 0.05),
            )
            feats_nonmask = (1 - t_prev) * x0_global + t_prev * noise_fixed
            # Hard switch (approach6-style), not soft blend.
            new_feats = torch.where(mask_bool, out.pred_x_prev.feats, feats_nonmask)
            sample_r = sample_r.replace(new_feats)
            del out

        with torch.no_grad():
            delta = sample_r.feats - x0_global
            dbg_inside = (mask_weight.squeeze(-1) > mask_threshold)
            n_active = int(dbg_inside.sum().item())
            # Per-voxel L2 norm over features
            per_voxel_l2 = delta.float().norm(dim=-1)
            inside_l2 = per_voxel_l2[dbg_inside]
            outside_l2 = per_voxel_l2[~dbg_inside]
            print(
                f"    SQ {sq_idx}: delta-norm  inside_mask mean={inside_l2.mean().item():.4f} "
                f"max={inside_l2.max().item() if n_active else 0:.4f}  | "
                f"outside mean={outside_l2.mean().item():.4f} max={outside_l2.max().item():.4f}  "
                f"| x0_global feat-norm mean={x0_global.float().norm(dim=-1).mean().item():.4f}"
            )
        # Hard replacement (approach6-style), not soft delta blend.
        result_feats = torch.where(mask_bool, sample_r.feats, result_feats)
        print(f"    SQ {sq_idx} done")

        if debug_dir is not None:
            _snapshot(f"after_SQ{sq_idx:02d}", result_feats)
            # Also snapshot the raw locally-refined branch (no blend, no
            # masking on the non-refined voxels). If this looks colored but
            # after_SQNN doesn't, the blend step is the culprit.
            _snapshot(f"local_only_SQ{sq_idx:02d}", sample_r.feats)

    # Apply pipeline normalization (mean/std) for decoder
    sample_out = sp.SparseTensor(feats=result_feats * std + mean, coords=coords)

    if debug_dir is not None:
        # Diagnostic snapshots decode many extra Gaussians; explicitly free
        # large temporaries and clear the allocator before the caller's
        # required final render.
        del noise_global, sample_g, x0_global, result_feats
        gc.collect()
        torch.cuda.empty_cache()

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


@torch.no_grad()
def sample_slat_extreme_v1(pipeline, coords, W, conds_dict, steps=15, cfg_strength=7.5,
                           debug_dir: Optional[str] = None):
    """Pre-91da279 multi-prompt fusion sampler (the one that actually produced
    distinctly-colored chair parts before Xander's May 11 rewrite).

    Math: at every denoising step, predict velocity with each SQ's prompt; fuse
    the per-step pred_x_prev's weighted by W; reconstruct velocity via dt and
    advance. CFG always on. No global cond, no rescale_t.
    """
    flow_model = pipeline.models['slat_flow_model_text']
    N, D = coords.shape[0], flow_model.in_channels
    device = pipeline.device

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)

    z_init = torch.randn(N, D, device=device)
    sample = sp.SparseTensor(feats=z_init, coords=coords)
    sampler = pipeline.slat_sampler
    t_seq = np.linspace(1, 0, steps + 1)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

    P = W.shape[1]
    print(f"Sampling extreme_v1 (pre-91da279): {P} prompts per step, {steps} steps")

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        _dbg_extr, _dbg_intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [0, np.pi / 2, np.pi, 3 * np.pi / 2], [0.35] * 4, 10, 8
        )

        def _snapshot(name, feats):
            try:
                snap = sp.SparseTensor(feats=feats * std + mean, coords=coords)
                gs = pipeline.decode_slat(snap, formats=["gaussian"])["gaussian"][0]
                scales = gs.get_scaling
                if not torch.isfinite(scales).all() or scales.max().item() > 10.0:
                    print(f"  [debug] snapshot {name}: degenerate Gaussian, skipping")
                    return
                frames = render_utils.render_frames(
                    gs, _dbg_extr, _dbg_intr,
                    {"resolution": 512, "bg_color": (255, 255, 255)},
                )["color"]
                for j, frame in enumerate(frames):
                    Image.fromarray(frame).save(os.path.join(debug_dir, f"{name}_view_{j}.png"))
                Image.fromarray(np.concatenate(frames, axis=1)).save(
                    os.path.join(debug_dir, f"{name}_grid.png")
                )
                print(f"  [debug] snapshot {name} saved")
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [debug] snapshot {name}: OOM ({e}); skipping")
            except Exception as e:
                print(f"  [debug] snapshot {name}: failed ({type(e).__name__}: {e})")
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    else:
        _snapshot = None

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        dt = t_prev - t
        feats_fused = torch.zeros_like(sample.feats)

        for sq_idx, cond in conds_dict.items():
            out = sampler.sample_once(
                flow_model, sample, t, t_prev, cond['cond'],
                cfg_strength=cfg_strength, neg_cond=cond.get('neg_cond'),
                cfg_interval=(0.0, 1.0),
            )
            mask = W[:, sq_idx:sq_idx + 1]
            feats_fused += mask * out.pred_x_prev.feats
            del out

        v_fused = (feats_fused - sample.feats) / dt
        sample = sample.replace(sample.feats + dt * v_fused)
        if step_idx % 3 == 0:
            print(f"    Step {step_idx}/{steps}")
        if _snapshot is not None and (step_idx == 0 or (step_idx + 1) % 3 == 0 or step_idx == steps - 1):
            _snapshot(f"step_{step_idx:02d}", sample.feats)

    return sp.SparseTensor(feats=sample.feats * std + mean, coords=coords)


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
