"""
Approach 7: Coupled Diffusion Sampling for Per-Superquadric Semantic Routing.

Builds on Approach 6 (per-SQ local prompts + hard spatial routing), but adds:
  (a) soft voxel-to-SQ weights (softmax over -dist/tau) instead of hard argmin,
  (b) an extra "global" denoising branch conditioned on the whole-object prompt,
  (c) a coupling term that, at every flow step, pulls each per-SQ predicted
      clean sample x_hat_0^i toward the global predicted clean sample x_hat_0^g,
      using a squared-L2 coupling energy U = -lambda/2 * ||x_i - x_g||^2 so the
      gradient is simply -lambda * (x_i - x_g).

Rationale: Approach 6's bleeding comes from each per-SQ branch denoising with
no knowledge of its neighbors; hard W masking after the fact cannot fix seams
whose features were already baked in inconsistent directions. Coupling adds
the missing cross-branch signal on predicted x_hat_0, which is the quantity
the paper identifies as the natural place to apply the coupling gradient.

Note: TRELLIS's SLAT sampler is rectified flow, not stochastic DDPM, so we
apply the coupling as a correction on pred_x_0 and recompute pred_x_prev via
the same Euler step. There is no noise-injection step to "reject" bad
couplings, so lambda tuning matters more here than in the original paper.
"""

import os
import sys
import argparse
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
from approach1_experiment import coords_to_world, compute_mesh_normalization, load_sq_params
from approach6_experiment import superquadric_radial_distance


def compute_soft_W(voxel_pos, sq_params, mesh_center, mesh_scale, tau=0.02):
    """
    Soft voxel-to-SQ weights: W[n, i] = softmax(-dist[n, i] / tau).
    tau controls sharpness; small tau -> approaches hard argmin, large tau -> uniform.
    """
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


@torch.no_grad()
def sample_slat_coupled(
    pipeline,
    coords,
    W,
    conds_dict,
    cond_global,
    steps=25,
    cfg_strength=7.5,
    lam=0.5,
    rescale_t=3.0,
    detail_t_threshold=0.5,
):
    """
    Coupled denoising with time-staged CFG negatives:
      Structural phase (t >= detail_t_threshold): local neg = global prompt
        -> branches inherit global structure, add only part-specific deviation.
      Detail phase    (t <  detail_t_threshold): local neg = null
        -> full local signal so color/material features actually form.
    Global branch always uses null negative (standard CFG).
    Coupling pulls local x_0 toward global x_0 by lam at every step.
    """
    flow_model = pipeline.models['slat_flow_model_text']
    N, D = coords.shape[0], flow_model.in_channels
    device = pipeline.device
    null_cond = pipeline.text_cond_model['null_cond']

    z_init = torch.randn(N, D, device=device)
    sample = sp.SparseTensor(feats=z_init, coords=coords)
    sample_g = sp.SparseTensor(feats=z_init.clone(), coords=coords)

    sampler = pipeline.slat_sampler
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

    P = W.shape[1]
    print(f"Sampling Coupled ({P} local + 1 global), lambda={lam}, detail_threshold={detail_t_threshold}")

    for step_idx, (t, t_prev) in enumerate(t_pairs):
        neg_local = cond_global['cond'] if t >= detail_t_threshold else null_cond

        # --- global branch: standard CFG with null negative ---
        out_g = sampler.sample_once(
            flow_model, sample_g, t, t_prev, cond_global['cond'],
            cfg_strength=cfg_strength, neg_cond=cond_global.get('neg_cond'),
            cfg_interval=(0.0, 0.95),
        )
        x0_g = out_g.pred_x_0.feats  # (N, D)

        # --- local branches: time-staged neg, coupling to x0_g ---
        feats_fused = torch.zeros_like(sample.feats)
        for sq_idx, cond in conds_dict.items():
            out_i = sampler.sample_once(
                flow_model, sample, t, t_prev, cond['cond'],
                cfg_strength=cfg_strength, neg_cond=neg_local,
                cfg_interval=(0.0, 0.95),
            )
            # Coupling on x_0: pull local x_0 toward global x_0 by lam.
            x0_i = out_i.pred_x_0.feats
            x0_i_coupled = x0_i + lam * (x0_g - x0_i)

            frac = (t - t_prev) / max(t, 1e-6)
            feats_i_prev = sample.feats + frac * (x0_i_coupled - sample.feats)

            mask = W[:, sq_idx:sq_idx + 1]
            feats_fused += mask * feats_i_prev
            del out_i

        sample = sample.replace(feats_fused)
        sample_g = sample_g.replace(out_g.pred_x_prev.feats)
        del out_g

        if step_idx % 5 == 0:
            print(f"    Step {step_idx}/{steps}")

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(device)
    return sample * std + mean


def run_experiment(
    sq_path="gui/superquadrics/chair_sq.npz",
    output_dir="approach7_results",
    steps=25,
    seed=42,
    lam=0.3,
    tau=0.02,
    cfg_strength=7.5,
):
    os.makedirs(output_dir, exist_ok=True)

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
        for i in range(P):
            if i not in local_prompts_text:
                local_prompts_text[i] = "a plain gray plastic chair part"

    global_structure_prompt = (
        "a minimalist chair with four thin legs, crossbars, "
        "a seat cushion, and a backrest"
    )

    print("2. Loading Pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts_text.items()}
    cond_global = pipeline.get_cond_text([global_structure_prompt])
    cond_struct = cond_global

    print("3. Sampling Base Structure...")
    torch.manual_seed(seed)
    coords = pipeline.sample_sparse_structure(
        cond_struct, num_samples=1, sampler_params={"steps": steps},
    )
    W = compute_soft_W(coords_to_world(coords), sq_params, mesh_center, mesh_scale, tau=tau)

    print(f"\n--- Running Experiment 7 (Coupled, lambda={lam}, tau={tau}) ---")
    torch.manual_seed(seed)
    slat = sample_slat_coupled(
        pipeline, coords, W, conds_local, cond_global,
        steps=steps, cfg_strength=cfg_strength, lam=lam,
    )
    gs = pipeline.decode_slat(slat, formats=['gaussian'])['gaussian'][0]

    print("Rendering...")
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2], [0.35] * 4, 10, 8,
    )
    frames = render_utils.render_frames(
        gs, extr, intr, {'resolution': 512, 'bg_color': (255, 255, 255)},
    )['color']

    row_img = np.concatenate(frames, axis=1)
    out_path = os.path.join(
        output_dir, f"coupled_lam{lam}_tau{tau}_chair.png",
    )
    Image.fromarray(row_img).save(out_path)
    print(f"\nDone! Coupled result saved to: {out_path}")

    with open(os.path.join(output_dir, "prompts.txt"), "w") as f:
        f.write(f"GLOBAL: {global_structure_prompt}\n\n")
        for k, v in local_prompts_text.items():
            f.write(f"SQ {k}: {v}\n")
        f.write(f"\nlambda={lam}  tau={tau}  cfg={cfg_strength}  steps={steps}  seed={seed}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sq-path", default="gui/superquadrics/chair_sq.npz")
    p.add_argument("--output-dir", default="approach7_results")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lam", type=float, default=0.3,
                   help="coupling strength in [0, 1]; 0 = Approach 6, 1 = global-only")
    p.add_argument("--tau", type=float, default=0.02,
                   help="soft-W temperature; small -> hard argmin, large -> uniform")
    p.add_argument("--cfg-strength", type=float, default=7.5)
    args = p.parse_args()

    run_experiment(
        sq_path=args.sq_path,
        output_dir=args.output_dir,
        steps=args.steps,
        seed=args.seed,
        lam=args.lam,
        tau=args.tau,
        cfg_strength=args.cfg_strength,
    )
