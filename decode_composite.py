import torch
import numpy as np

from trellis.modules import sparse as sp
from trellis.pipelines.samplers.flow_euler import FlowEulerSampler


def superquadric_radial_distance(x_local, semi_axes, eps):
    e1, e2 = eps[0].clamp(min=0.01), eps[1].clamp(min=0.01)
    ax = semi_axes[0].clamp(min=1e-6)
    ay = semi_axes[1].clamp(min=1e-6)
    az = semi_axes[2].clamp(min=1e-6)
    x, y, z = x_local[:, 0], x_local[:, 1], x_local[:, 2]
    f = (
        (torch.abs(x / ax) ** (2 / e2) + torch.abs(y / ay) ** (2 / e2)) ** (e2 / e1)
        + torch.abs(z / az) ** (2 / e1)
    )
    f = f.clamp(min=1e-12)
    return torch.norm(x_local, dim=-1) * torch.abs(1.0 - f ** (-e1 / 2.0))


def _sq_distance_matrix(positions, sq_params, mesh_center, mesh_scale):
    device = positions.device
    n, p = positions.shape[0], len(sq_params)
    dist = torch.zeros(n, p, device=device)
    m_center = torch.tensor(mesh_center, device=device).float()
    for i, sq in enumerate(sq_params):
        center = (torch.tensor(sq["translation"], device=device).float() - m_center) * mesh_scale
        rot = torch.tensor(sq["rotation"], device=device).float()
        scale = torch.tensor(sq["scale"], device=device).float() * mesh_scale
        shape = torch.tensor(sq["shape"], device=device).float()
        x_loc = (positions - center.unsqueeze(0)) @ rot
        dist[:, i] = superquadric_radial_distance(x_loc, scale, shape)
    return dist


def _coords_to_world(coords, grid_size=64):
    return (coords[:, 1:4].float() + 0.5) / grid_size - 0.5


def _voxel_masks_per_prompt(coords, sq_params, mesh_center, mesh_scale,
                            prompt_to_sqs, soft_tau=None):
    """For each unique prompt key, return an (N, 1) mask over voxels that
    sums (across prompts) to 1 on every assigned voxel.
    """
    positions = _coords_to_world(coords).to(coords.device)
    dist = _sq_distance_matrix(positions, sq_params, mesh_center, mesh_scale)
    if soft_tau is not None:
        w = torch.softmax(-dist / soft_tau, dim=1)
    else:
        idx = dist.argmin(dim=1)
        w = torch.zeros_like(dist)
        w.scatter_(1, idx.unsqueeze(1), 1.0)
    return {key: w[:, sqs].sum(dim=1, keepdim=True) for key, sqs in prompt_to_sqs.items()}


def _predict_v(sampler, flow_model, x_t, t_float, cond_tensor):
    """Single conditional velocity prediction, bypassing any CFG mixin."""
    return FlowEulerSampler._inference_model(sampler, flow_model, x_t, t_float, cond_tensor)


@torch.no_grad()
def sample_composite_slat(
    pipeline,
    coords,
    conds_local,
    cond_global,
    sq_params,
    mesh_center,
    mesh_scale,
    steps=25,
    cfg_strength=7.5,
    local_cfg_strength=15.0,
    rescale_t=3.0,
    cfg_interval=(0.5, 0.95),
    soft_tau=None,
):
    """Compositional CFG with per-region strength. At each denoising step we
    predict velocity once per unique prompt plus once for the negative prompt.
    For each region we form a standard CFG velocity using that region's strength
    (local_cfg_strength for local prompts, cfg_strength for the global prompt),
    then blend those regional CFG velocities in voxel space using the SQ masks.

    The negative pass is shared across regions, so total forward passes per step
    is (#unique_prompts + 1). All regions share one noise trajectory → coherent
    SLAT, no slice-and-merge.

    Per region: v_cfg_i = (1+s_i) * v_cond_i - s_i * v_neg.
    Combined:  v = Σ mask_i * v_cfg_i.
    """
    device = pipeline.device
    flow_model = pipeline.models["slat_flow_model_text"]
    sampler = pipeline.slat_sampler
    std = torch.tensor(pipeline.slat_normalization["std"])[None].to(device)
    mean = torch.tensor(pipeline.slat_normalization["mean"])[None].to(device)
    neg_cond_tensor = cond_global["neg_cond"]
    global_key = id(cond_global)

    prompt_to_sqs = {}
    cond_for_prompt = {}
    for sq_idx, cond in conds_local.items():
        key = id(cond)
        prompt_to_sqs.setdefault(key, []).append(sq_idx)
        cond_for_prompt[key] = cond

    strength_for_prompt = {
        key: (cfg_strength if key == global_key else local_cfg_strength)
        for key in cond_for_prompt
    }

    masks = _voxel_masks_per_prompt(
        coords, sq_params, mesh_center, mesh_scale, prompt_to_sqs, soft_tau=soft_tau,
    )

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

    noise = torch.randn(coords.shape[0], flow_model.in_channels, device=device)
    sample = sp.SparseTensor(feats=noise, coords=coords)

    print(
        f"[decode_composite] compositional CFG: {len(cond_for_prompt)} unique prompts, "
        f"{steps} steps, cfg(global)={cfg_strength}, cfg(local)={local_cfg_strength}, "
        f"cfg_interval={cfg_interval}, soft_tau={soft_tau}"
    )

    for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
        cfg_on = cfg_interval[0] <= t <= cfg_interval[1]

        v_pos_blend = torch.zeros_like(sample.feats)
        neg_weight = torch.zeros_like(sample.feats)
        for key, mask in masks.items():
            v_i = _predict_v(sampler, flow_model, sample, t, cond_for_prompt[key]["cond"])
            s_i = strength_for_prompt[key] if cfg_on else 0.0
            v_pos_blend = v_pos_blend + mask * (1.0 + s_i) * v_i.feats
            if s_i > 0:
                neg_weight = neg_weight + mask * s_i
            del v_i

        if cfg_on and neg_weight.abs().max() > 0:
            v_neg = _predict_v(sampler, flow_model, sample, t, neg_cond_tensor)
            v_combined = v_pos_blend - neg_weight * v_neg.feats
            del v_neg
        else:
            v_combined = v_pos_blend

        sample = sample.replace(sample.feats - (t - t_prev) * v_combined)

    return sp.SparseTensor(feats=sample.feats * std + mean, coords=coords)


@torch.no_grad()
def decode_composite_gaussian(
    pipeline,
    coords,
    conds_local,
    cond_global,
    sq_params,
    mesh_center,
    mesh_scale,
    steps=25,
    cfg_strength=7.5,
    rescale_t=3.0,
    local_cfg_strength=15.0,
    cfg_interval=(0.5, 0.95),
    soft_tau=None,
):
    """Sample one composite SLAT via compositional CFG and decode it. Returns
    (gaussian, mesh) decoded from the same SLAT — appearance varies by SQ region,
    geometry stays globally coherent because all regions share one noise tensor
    and one sampling trajectory. local_cfg_strength controls how hard each local
    prompt is pushed within its SQ region (independent of the global cfg_strength).
    """
    slat = sample_composite_slat(
        pipeline, coords, conds_local, cond_global,
        sq_params, mesh_center, mesh_scale,
        steps=steps, cfg_strength=cfg_strength,
        local_cfg_strength=local_cfg_strength,
        rescale_t=rescale_t, cfg_interval=cfg_interval, soft_tau=soft_tau,
    )
    out = pipeline.decode_slat(slat, formats=["gaussian", "mesh"])
    return out["gaussian"][0], out["mesh"][0]
