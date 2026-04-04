"""
Approach 1: Constrained Denoising via Superquadric Projection

Interleaves standard TRELLIS denoising steps with a projection onto the
superquadric subspace, so that voxels belonging to the same primitive share
coherent appearance features.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from easydict import EasyDict as edict

os.environ['SPCONV_ALGO'] = 'native'
sys.path.insert(0, os.path.dirname(__file__))

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils, postprocessing_utils


# ---------------------------------------------------------------------------
# Superquadric weight computation
# ---------------------------------------------------------------------------

def superquadric_inside_outside(x_local: torch.Tensor,
                                 semi_axes: torch.Tensor,
                                 eps: torch.Tensor) -> torch.Tensor:
    """
    Compute the inside-outside function f(x) for a canonically-oriented
    superquadric (Eq. 3 in the proposal).

    Args:
        x_local: (N, 3) points in the superquadric's local frame.
        semi_axes: (3,) semi-axis lengths (a1, a2, a3).
        eps: (2,) shape exponents (epsilon_1, epsilon_2).

    Returns:
        f: (N,) inside-outside values. f < 1 inside, f > 1 outside.
    """
    e1, e2 = eps[0].clamp(min=0.01), eps[1].clamp(min=0.01)
    a1, a2, a3 = semi_axes[0].clamp(min=1e-6), semi_axes[1].clamp(min=1e-6), semi_axes[2].clamp(min=1e-6)

    x, y, z = x_local[:, 0], x_local[:, 1], x_local[:, 2]

    term_xy = (torch.abs(x / a1) ** (2.0 / e2) +
               torch.abs(y / a2) ** (2.0 / e2))
    term_xy = term_xy.clamp(min=1e-12)
    f = term_xy ** (e2 / e1) + torch.abs(z / a3) ** (2.0 / e1)
    return f


def superquadric_radial_distance(x_local: torch.Tensor,
                                  semi_axes: torch.Tensor,
                                  eps: torch.Tensor) -> torch.Tensor:
    """
    Radial distance from points to the superquadric surface (Eq. 2).

    d_r(x) = |x| * |1 - f(x)^{-eps1/2}|
    """
    e1 = eps[0].clamp(min=0.01)
    f = superquadric_inside_outside(x_local, semi_axes, eps)
    f = f.clamp(min=1e-12)
    norm = torch.norm(x_local, dim=-1).clamp(min=1e-12)
    d_r = norm * torch.abs(1.0 - f ** (-e1 / 2.0))
    return d_r


def compute_weight_matrix(voxel_positions: torch.Tensor,
                          sq_params: List[Dict],
                          mesh_center: np.ndarray,
                          mesh_scale: float,
                          tau: float = 0.02) -> torch.Tensor:
    """
    Compute the N x P weight matrix W where W[j, i] encodes how strongly
    voxel j is influenced by superquadric i (Eq. 4, normalized).

    Args:
        voxel_positions: (N, 3) world positions of voxels in [-0.5, 0.5].
        sq_params: list of P dicts, each with 'scale', 'shape', 'rotation', 'translation'.
        mesh_center: (3,) center used for normalization.
        mesh_scale: scalar used for normalization.
        tau: temperature for the exponential kernel.

    Returns:
        W: (N, P) normalized weight matrix.
    """
    device = voxel_positions.device
    N = voxel_positions.shape[0]
    P = len(sq_params)
    W = torch.zeros(N, P, device=device, dtype=torch.float32)

    for i, sq in enumerate(sq_params):
        center = torch.tensor(sq['translation'], dtype=torch.float32, device=device)
        rotation = torch.tensor(sq['rotation'], dtype=torch.float32, device=device)
        semi_axes = torch.tensor(sq['scale'], dtype=torch.float32, device=device)
        shape_exp = torch.tensor(sq['shape'], dtype=torch.float32, device=device)

        center_norm = (center - torch.tensor(mesh_center, dtype=torch.float32, device=device)) * mesh_scale
        semi_axes_norm = semi_axes * mesh_scale

        x_centered = voxel_positions - center_norm.unsqueeze(0)
        x_local = (rotation.T @ x_centered.T).T  # (N, 3)

        d_r = superquadric_radial_distance(x_local, semi_axes_norm, shape_exp)
        W[:, i] = torch.exp(-d_r / tau)

    W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
    return W


def coords_to_world_positions(coords: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
    """Convert integer grid coords (batch, x, y, z) to world positions in [-0.5, 0.5]."""
    xyz = coords[:, 1:4].float()
    world_pos = (xyz + 0.5) / grid_size - 0.5
    return world_pos


# ---------------------------------------------------------------------------
# Projection step
# ---------------------------------------------------------------------------

def project_onto_sq_subspace(z: torch.Tensor,
                              W: torch.Tensor,
                              ridge: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project voxel latents onto the superquadric subspace (Eq. 5-6).

    Solves s* = argmin_s ||z - W s||^2  (per latent dimension).
    Returns projected voxel latents z_bar = W s*.

    Args:
        z: (N, D) voxel latent features.
        W: (N, P) weight matrix.
        ridge: regularization for numerical stability.

    Returns:
        z_bar: (N, D) projected voxel latents.
        s_star: (P, D) superquadric feature vectors.
    """
    P = W.shape[1]
    WtW = W.T @ W + ridge * torch.eye(P, device=W.device, dtype=W.dtype)
    Wtz = W.T @ z
    s_star = torch.linalg.solve(WtW, Wtz)
    z_bar = W @ s_star
    return z_bar, s_star


# ---------------------------------------------------------------------------
# Modified sampler with projection
# ---------------------------------------------------------------------------

def sample_slat_with_projection(
    pipeline: TrellisTextTo3DPipeline,
    cond: dict,
    coords: torch.Tensor,
    W: torch.Tensor,
    sampler_params: dict = {},
    project_every: int = 1,
    blend_alpha: float = 1.0,
    project_after_frac: float = 0.0,
    final_project: bool = True,
) -> sp.SparseTensor:
    """
    Sample structured latent with superquadric projection interleaved
    between Euler denoising steps (Approach 1).

    Args:
        pipeline: the TRELLIS pipeline.
        cond: conditioning dict with 'cond' and 'neg_cond'.
        coords: (N, 4) voxel coordinates.
        W: (N, P) precomputed weight matrix.
        sampler_params: sampler parameters.
        project_every: project every k steps (1 = every step).
        blend_alpha: mixing ratio in [0, 1]. 0 = no projection (baseline),
                     1 = full projection (hard). Values in between blend:
                     z_mixed = alpha * z_projected + (1-alpha) * z_original.
        project_after_frac: only start projecting after this fraction of
                            steps has elapsed (0.0 = from start, 0.5 = last half).
        final_project: whether to do a hard projection on the final output.

    Returns:
        slat: SparseTensor with denoised latents.
    """
    if cond['cond'].shape[-1] == 768:
        flow_model = pipeline.models['slat_flow_model_text']
    else:
        flow_model = pipeline.models['slat_flow_model_image']

    noise = sp.SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model.in_channels).to(pipeline.device),
        coords=coords,
    )

    params = {**pipeline.slat_sampler_params, **sampler_params}
    sampler = pipeline.slat_sampler

    steps = params.get('steps', 25)
    rescale_t = params.get('rescale_t', 1.0)
    cfg_strength = params.get('cfg_strength', 7.5)
    cfg_interval = params.get('cfg_interval', (0.0, 1.0))
    neg_cond = cond.get('neg_cond')

    sample = noise
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

    args = {
        'neg_cond': neg_cond,
        'cfg_strength': cfg_strength,
        'cfg_interval': cfg_interval,
    }

    start_step = int(project_after_frac * steps)
    label = f"alpha={blend_alpha}, every={project_every}, start@step{start_step}/{steps}"
    print(f"Running Approach 1 sampling: {steps} steps, {label}")

    for step_idx, (t, t_prev) in enumerate(t_pairs):
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

        out = sampler.sample_once(flow_model, sample, t, t_prev, cond['cond'], **args)
        sample = out.pred_x_prev

        if step_idx % 5 == 0:
            print(f"  Step {step_idx}/{steps}, t={t:.4f}")

    if final_project:
        z_bar, s_star = project_onto_sq_subspace(sample.feats, W)
        sample = sample.replace(z_bar)
        print(f"  Final SQ features shape: {s_star.shape}")

    std = torch.tensor(pipeline.slat_normalization['std'])[None].to(sample.device)
    mean = torch.tensor(pipeline.slat_normalization['mean'])[None].to(sample.device)
    sample = sample * std + mean

    return sample


# ---------------------------------------------------------------------------
# Helper: compute mesh normalization from SQ params
# ---------------------------------------------------------------------------

def compute_mesh_normalization(sq_params: List[Dict]) -> Tuple[np.ndarray, float]:
    """Reproduce the center/scale normalization from gui_text_image.py's generate()."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gui'))
    from gui_text_image import add_superquadric_compact_rot_mat

    all_vertices = []
    for sq in sq_params:
        vertices, _ = add_superquadric_compact_rot_mat(
            sq['scale'], sq['shape'], sq['translation'], sq['rotation'], resolution=100)
        all_vertices.append(vertices)
    all_vertices = np.concatenate(all_vertices, axis=0)
    aabb = np.stack([all_vertices.min(0), all_vertices.max(0)])
    center = (aabb[0] + aabb[1]) / 2
    scale = 1.0 / (aabb[1] - aabb[0]).max()
    return center, scale


def load_sq_params(npz_path: str) -> List[Dict]:
    """Load superquadric parameters from an npz file."""
    data = np.load(npz_path)
    params = []
    for k in range(data['scales'].shape[0]):
        params.append({
            'scale': data['scales'][k],
            'shape': data['shapes'][k],
            'rotation': data['rotations'][k],
            'translation': data['translations'][k],
        })
    return params


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
    project_every: int = 1,
    output_dir: str = "approach1_results",
):
    os.makedirs(output_dir, exist_ok=True)

    # --- Load SQ params ---
    print("Loading superquadric parameters...")
    sq_params = load_sq_params(sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)
    print(f"  {len(sq_params)} superquadrics, mesh_center={mesh_center}, mesh_scale={mesh_scale:.4f}")

    # --- Build SQ mesh and save for the pipeline ---
    import open3d as o3d
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gui'))
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

    # --- Sample structure (shared between baseline and approach 1) ---
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

    # --- Compute voxel world positions ---
    voxel_positions = coords_to_world_positions(coords)
    print(f"  Voxel position range: [{voxel_positions.min():.3f}, {voxel_positions.max():.3f}]")

    # --- Compute weight matrix ---
    print("Computing superquadric weight matrix...")
    W = compute_weight_matrix(voxel_positions, sq_params, mesh_center, mesh_scale, tau=tau)
    print(f"  W shape: {W.shape}")
    print(f"  W stats: min={W.min():.6f}, max={W.max():.6f}, mean={W.mean():.6f}")

    dominant = W.argmax(dim=1)
    for i in range(len(sq_params)):
        count = (dominant == i).sum().item()
        print(f"    SQ {i}: {count} dominant voxels")

    # --- Define experiment configurations ---
    configs = [
        {"name": "Baseline (TRELLIS)",              "blend_alpha": 0.0,  "project_every": 1, "project_after_frac": 0.0, "final_project": False},
        {"name": "alpha=0.1, every step",           "blend_alpha": 0.1,  "project_every": 1, "project_after_frac": 0.0, "final_project": True},
        {"name": "alpha=0.3, every step",           "blend_alpha": 0.3,  "project_every": 1, "project_after_frac": 0.0, "final_project": True},
        {"name": "alpha=0.5, every step",           "blend_alpha": 0.5,  "project_every": 1, "project_after_frac": 0.0, "final_project": True},
        {"name": "alpha=1.0, last 50% only",        "blend_alpha": 1.0,  "project_every": 1, "project_after_frac": 0.5, "final_project": True},
        {"name": "final project only (no interleave)", "blend_alpha": 0.0, "project_every": 1, "project_after_frac": 1.0, "final_project": True},
    ]

    all_views = []
    labels = []

    from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    pitchs = [0.35] * 4
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, 10, 8)

    def render_views(gaussian):
        frames = render_utils.render_frames(
            gaussian, extrinsics, intrinsics,
            {'resolution': 512, 'bg_color': (255, 255, 255)},
            verbose=False
        )
        return frames['color']

    for cfg in configs:
        name = cfg["name"]
        print(f"\n=== {name} ===")
        torch.manual_seed(seed)

        if cfg["blend_alpha"] == 0.0 and not cfg["final_project"]:
            slat = pipeline.sample_slat(cond, coords)
        else:
            slat = sample_slat_with_projection(
                pipeline, cond, coords, W,
                sampler_params={},
                project_every=cfg["project_every"],
                blend_alpha=cfg["blend_alpha"],
                project_after_frac=cfg["project_after_frac"],
                final_project=cfg["final_project"],
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

    # --- Create comparison grid ---
    print("\nCreating comparison grid...")
    n_views = len(all_views[0])
    img_h, img_w = all_views[0][0].shape[:2]
    n_rows = len(configs)
    label_height = 40

    from PIL import ImageDraw, ImageFont
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

    # --- Save superquadric assignment visualization ---
    print("Creating superquadric assignment visualization...")
    create_sq_assignment_viz(coords, W, sq_params, output_dir)

    print(f"\nDone! Results saved to {output_dir}/")
    return grid_path


def create_sq_assignment_viz(coords: torch.Tensor, W: torch.Tensor,
                              sq_params: List[Dict], output_dir: str):
    """
    Visualize which superquadric dominates each voxel using color-coded
    point clouds rendered from multiple views.
    """
    P = len(sq_params)
    dominant = W.argmax(dim=1).cpu().numpy()

    cmap = np.array([
        [228, 26, 28],   [55, 126, 184],  [77, 175, 74],
        [152, 78, 163],  [255, 127, 0],   [255, 255, 51],
        [166, 86, 40],   [247, 129, 191], [153, 153, 153],
        [0, 0, 0],       [141, 211, 199], [255, 255, 179],
        [190, 186, 218], [251, 128, 114], [128, 177, 211],
        [253, 180, 98],
    ], dtype=np.uint8)

    colors = cmap[dominant % len(cmap)]

    import open3d as o3d
    xyz = coords[:, 1:4].cpu().float().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(os.path.join(output_dir, "sq_assignment.ply"), pcd)

    from PIL import ImageDraw, ImageFont
    fig_data = np.zeros((512, 512, 3), dtype=np.uint8)
    fig_data[:] = 255
    legend_img = Image.fromarray(fig_data)
    draw = ImageDraw.Draw(legend_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    draw.text((10, 10), "Superquadric Assignment", fill=(0, 0, 0), font=font)
    for i in range(P):
        y = 40 + i * 25
        c = tuple(cmap[i % len(cmap)])
        count = (dominant == i).sum()
        draw.rectangle([10, y, 30, y + 18], fill=c)
        draw.text((40, y), f"SQ {i}: {count} voxels", fill=(0, 0, 0), font=font)

    legend_img.save(os.path.join(output_dir, "sq_assignment_legend.png"))
    print(f"  Saved assignment point cloud and legend to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Approach 1 experiment")
    parser.add_argument("--sq-path", default="gui/superquadrics/chair_sq.npz")
    parser.add_argument("--prompt", default="a wooden chair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.02,
                        help="Temperature for distance-based weighting")
    parser.add_argument("--t0-idx", type=int, default=6)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg-strength", type=float, default=7.5)
    parser.add_argument("--project-every", type=int, default=1)
    parser.add_argument("--output-dir", default="approach1_results")
    args = parser.parse_args()

    grid_path = run_experiment(
        sq_path=args.sq_path,
        text_prompt=args.prompt,
        seed=args.seed,
        tau=args.tau,
        t0_idx=args.t0_idx,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        project_every=args.project_every,
        output_dir=args.output_dir,
    )
