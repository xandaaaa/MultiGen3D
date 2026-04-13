"""
Approach 1 (revised): Part-Level Appearance Transplant via Superquadric Assignment

Hypothesis: SLAT features encode local appearance independently at each voxel.
We can replace features for voxels belonging to one superquadric with features
from a different-style generation, achieving part-level appearance editing while
preserving overall geometry.

Experiment:
  - Generate structure with spatial control (superquadrics fix active voxel coords).
  - Generate SLAT_A (base style) and SLAT_B (target style) on the SAME coords.
  - Hard-assign each voxel to its "most containing" superquadric (argmin of
    inside-outside function across all SQs).
  - For each SQ i: build a mixed SLAT where SQ_i voxels carry SLAT_B features,
    all other voxels keep SLAT_A features.
  - Decode all variants and compare in a single grid.

What to look for: if the SQ_i-replaced render shows style B only in the spatial
region of SQ_i while other parts retain style A, the SLAT is spatially
decomposable and part-level appearance editing is feasible.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple

os.environ['SPCONV_ALGO'] = 'native'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.modules import sparse as sp
from trellis.utils import render_utils


# ---------------------------------------------------------------------------
# Superquadric inside-outside function
# ---------------------------------------------------------------------------

def sq_inside_outside(x_local: torch.Tensor,
                      semi_axes: torch.Tensor,
                      eps: torch.Tensor) -> torch.Tensor:
    """
    F(x) < 1 → x is inside the superquadric.
    F(x) = 1 → x is on the surface.
    F(x) > 1 → x is outside.
    """
    e1 = eps[0].clamp(min=0.01)
    e2 = eps[1].clamp(min=0.01)
    a1 = semi_axes[0].clamp(min=1e-6)
    a2 = semi_axes[1].clamp(min=1e-6)
    a3 = semi_axes[2].clamp(min=1e-6)
    x, y, z = x_local[:, 0], x_local[:, 1], x_local[:, 2]
    xy = (torch.abs(x / a1) ** (2.0 / e2) +
          torch.abs(y / a2) ** (2.0 / e2)).clamp(min=1e-12)
    return xy ** (e2 / e1) + torch.abs(z / a3) ** (2.0 / e1)


def hard_assign_voxels(voxel_positions: torch.Tensor,
                       sq_params: List[Dict],
                       mesh_center: np.ndarray,
                       mesh_scale: float) -> torch.Tensor:
    """
    Assign each voxel to the superquadric it is "most inside" (argmin of F).

    Returns:
        assignment: (N,) int64 tensor of SQ indices.
    """
    device = voxel_positions.device
    N = voxel_positions.shape[0]
    P = len(sq_params)
    F = torch.zeros(N, P, device=device, dtype=torch.float32)

    mc = torch.tensor(mesh_center, dtype=torch.float32, device=device)

    for i, sq in enumerate(sq_params):
        center    = torch.tensor(sq['translation'], dtype=torch.float32, device=device)
        rotation  = torch.tensor(sq['rotation'],    dtype=torch.float32, device=device)
        semi_axes = torch.tensor(sq['scale'],       dtype=torch.float32, device=device)
        shape_exp = torch.tensor(sq['shape'],       dtype=torch.float32, device=device)

        center_n    = (center - mc) * mesh_scale
        semi_axes_n = semi_axes * mesh_scale

        x_local = (rotation.T @ (voxel_positions - center_n).T).T
        F[:, i] = sq_inside_outside(x_local, semi_axes_n, shape_exp)

    return F.argmin(dim=1)   # (N,) each voxel goes to its "most containing" SQ


def coords_to_world(coords: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
    """Integer grid coords (N, 4) → world positions in [-0.5, 0.5]³."""
    return (coords[:, 1:4].float() + 0.5) / grid_size - 0.5


# ---------------------------------------------------------------------------
# Feature transplant
# ---------------------------------------------------------------------------

def transplant_features(slat_a: sp.SparseTensor,
                        slat_b: sp.SparseTensor,
                        assignment: torch.Tensor,
                        sq_idx: int) -> sp.SparseTensor:
    """
    Return a new SparseTensor with the same coords as slat_a, but where
    voxels assigned to sq_idx carry slat_b features and all others keep
    slat_a features.
    """
    mask = (assignment == sq_idx)
    feats = slat_a.feats.clone()
    feats[mask] = slat_b.feats[mask]
    return sp.SparseTensor(feats=feats, coords=slat_a.coords)


# ---------------------------------------------------------------------------
# Mesh / SQ helpers
# ---------------------------------------------------------------------------

def compute_mesh_normalization(sq_params: List[Dict]) -> Tuple[np.ndarray, float]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gui'))
    from gui_text_image import add_superquadric_compact_rot_mat
    all_verts = []
    for sq in sq_params:
        v, _ = add_superquadric_compact_rot_mat(
            sq['scale'], sq['shape'], sq['translation'], sq['rotation'], resolution=50)
        all_verts.append(v)
    all_verts = np.concatenate(all_verts, axis=0)
    aabb   = np.stack([all_verts.min(0), all_verts.max(0)])
    center = (aabb[0] + aabb[1]) / 2
    scale  = 1.0 / (aabb[1] - aabb[0]).max()
    return center, scale


def load_sq_params(npz_path: str) -> List[Dict]:
    data = np.load(npz_path)
    return [{'scale': data['scales'][k], 'shape': data['shapes'][k],
             'rotation': data['rotations'][k], 'translation': data['translations'][k]}
            for k in range(data['scales'].shape[0])]


def build_normalized_sq_mesh(sq_params, mesh_center, mesh_scale):
    """Build an open3d mesh of all SQs normalized to the pipeline's coord space."""
    import open3d as o3d
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gui'))
    from gui_text_image import add_superquadric_compact_rot_mat
    from gui.utils import merge_meshes
    meshes = []
    for sq in sq_params:
        v, f = add_superquadric_compact_rot_mat(
            sq['scale'], sq['shape'], sq['translation'], sq['rotation'], resolution=50)
        m = o3d.geometry.TriangleMesh()
        m.vertices  = o3d.utility.Vector3dVector(v)
        m.triangles = o3d.utility.Vector3iVector(f)
        meshes.append(m)
    merged = merge_meshes(meshes)
    merged.translate(-mesh_center)
    merged.scale(mesh_scale, (0, 0, 0))
    return merged


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

CAMERAS = dict(
    yaws   = [0, np.pi / 2, np.pi, 3 * np.pi / 2],
    pitchs = [0.35] * 4,
    r      = 10,
    fov    = 8,
)


def get_cameras():
    from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(
        CAMERAS['yaws'], CAMERAS['pitchs'], CAMERAS['r'], CAMERAS['fov'])


def render_gs(pipeline, slat, extrinsics, intrinsics, res=512):
    gs    = pipeline.decode_slat(slat, formats=['gaussian'])['gaussian'][0]
    views = render_utils.render_frames(
        gs, extrinsics, intrinsics,
        {'resolution': res, 'bg_color': (255, 255, 255)}, verbose=False
    )['color']
    del gs; torch.cuda.empty_cache()
    return views   # list of np.ndarray (H, W, 3) uint8


def render_sq_mesh_normals(o3d_mesh, extrinsics, intrinsics, res=512):
    from trellis.representations.mesh.cube2mesh import MeshExtractResult
    verts = torch.tensor(np.asarray(o3d_mesh.vertices), dtype=torch.float32).cuda()
    faces = torch.tensor(np.asarray(o3d_mesh.triangles), dtype=torch.int64).cuda()
    return render_utils.render_frames(
        MeshExtractResult(vertices=verts, faces=faces),
        extrinsics, intrinsics, {'resolution': res}, verbose=False
    )['normal']


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

SQ_COLORS = np.array([
    [228,  26,  28], [55, 126, 184], [ 77, 175,  74], [152,  78, 163],
    [255, 127,   0], [166,  86,  40], [247, 129, 191], [153, 153, 153],
    [  0,   0,   0], [141, 211, 199],
], dtype=np.uint8)


def save_sq_assignment_viz(coords: torch.Tensor,
                           assignment: torch.Tensor,
                           n_sqs: int,
                           output_path: str):
    """
    Three-panel scatter plot (XY, XZ, YZ projections) of active voxels
    coloured by their assigned superquadric.
    """
    pts = coords[:, 1:4].cpu().float().numpy()   # (N, 3)
    asgn = assignment.cpu().numpy()
    colors = SQ_COLORS[asgn % len(SQ_COLORS)] / 255.0  # (N, 3) float

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    projections = [
        (0, 1, 'X', 'Y', 'Front  (XY)'),
        (0, 2, 'X', 'Z', 'Side   (XZ)'),
        (1, 2, 'Y', 'Z', 'Top    (YZ)'),
    ]
    for ax, (xi, yi, xl, yl, title) in zip(axes, projections):
        ax.scatter(pts[:, xi], pts[:, yi], c=colors, s=1.5, linewidths=0)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title); ax.set_aspect('equal')

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=SQ_COLORS[i % len(SQ_COLORS)] / 255.0,
                   markersize=8, label=f'SQ {i}')
        for i in range(n_sqs)
    ]
    fig.legend(handles=handles, loc='lower center', ncol=n_sqs,
               frameon=False, fontsize=9)
    fig.suptitle('Voxel → Superquadric Assignment', fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_path}")


def save_comparison_grid(rows: List[List[np.ndarray]],
                         labels: List[str],
                         output_path: str):
    """
    Save a grid: one row per label, one column per camera view.
    rows[i] is a list of (H, W, 3) uint8 images.
    """
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    n_rows = len(rows)
    n_cols = len(rows[0])
    h, w   = rows[0][0].shape[:2]
    lh     = 36   # label header height

    canvas = np.ones((n_rows * (h + lh), n_cols * w, 3), dtype=np.uint8) * 240
    img    = Image.fromarray(canvas)
    draw   = ImageDraw.Draw(img)

    for r, (views, label) in enumerate(zip(rows, labels)):
        y0 = r * (h + lh)
        draw.text((6, y0 + 8), label, fill=(30, 30, 30), font=font)
        for c, view in enumerate(views):
            x0 = c * w
            img.paste(Image.fromarray(view), (x0, y0 + lh))

    img.save(output_path)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    sq_path:      str   = "gui/superquadrics/chair_sq.npz",
    prompt_a:     str   = "a wooden chair",
    prompt_b:     str   = "a blue metal chair",
    seed:         int   = 42,
    t0_idx:       int   = 6,
    steps:        int   = 12,
    cfg_strength: float = 7.5,
    output_dir:   str   = "approach1_results",
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- SQ params --------------------------------------------------------
    print("Loading superquadric parameters...")
    sq_params   = load_sq_params(sq_path)
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)
    P = len(sq_params)
    print(f"  {P} superquadrics")

    # ---- Normalised SQ mesh (for spatial control and rendering) -----------
    import open3d as o3d
    sq_mesh = build_normalized_sq_mesh(sq_params, mesh_center, mesh_scale)
    sq_mesh_path = os.path.join(output_dir, "sq_mesh.ply")
    o3d.io.write_triangle_mesh(sq_mesh_path, sq_mesh)

    # ---- Pipeline ---------------------------------------------------------
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    extrinsics, intrinsics = get_cameras()

    # ---- Stage 1: sample structure with spatial control -------------------
    print("Stage 1: sampling sparse structure...")
    cond_a = pipeline.get_cond_text([prompt_a])
    torch.manual_seed(seed)
    spatial_latent = pipeline.encode_spatial_control(sq_mesh_path)
    cond_ctrl = {**cond_a, 'control': spatial_latent}
    coords = pipeline.sample_sparse_structure(
        cond_ctrl, num_samples=1,
        sampler_params={"steps": steps, "cfg_strength": cfg_strength,
                        "t0_idx_value": t0_idx,
                        "spatial_control_mesh_path": sq_mesh_path},
    )
    print(f"  {coords.shape[0]} active voxels")

    # ---- Hard voxel-to-SQ assignment --------------------------------------
    print("Computing voxel → SQ assignment...")
    voxel_pos  = coords_to_world(coords).to(pipeline.device)
    assignment = hard_assign_voxels(voxel_pos, sq_params, mesh_center, mesh_scale)
    for i in range(P):
        print(f"  SQ {i}: {(assignment == i).sum().item()} voxels")

    # ---- Stage 2: sample appearance for both styles -----------------------
    print(f"Stage 2: sampling SLAT_A  ({prompt_a})...")
    torch.manual_seed(seed)
    slat_a = pipeline.sample_slat(cond_a, coords)

    print(f"Stage 2: sampling SLAT_B  ({prompt_b})...")
    cond_b = pipeline.get_cond_text([prompt_b])
    torch.manual_seed(seed)
    slat_b = pipeline.sample_slat(cond_b, coords)

    # ---- Render all variants ----------------------------------------------
    rows, labels = [], []

    print("Rendering SQ mesh normals...")
    rows.append(render_sq_mesh_normals(sq_mesh, extrinsics, intrinsics))
    labels.append("Input superquadrics (shape template)")

    print("Rendering SLAT_A baseline...")
    rows.append(render_gs(pipeline, slat_a, extrinsics, intrinsics))
    labels.append(f"Style A – {prompt_a}")

    print("Rendering SLAT_B baseline...")
    rows.append(render_gs(pipeline, slat_b, extrinsics, intrinsics))
    labels.append(f"Style B – {prompt_b}")

    print("Rendering per-SQ transplants (B features into A)...")
    for i in range(P):
        n_vox = (assignment == i).sum().item()
        print(f"  SQ {i} ({n_vox} voxels)...")
        slat_mix = transplant_features(slat_a, slat_b, assignment, sq_idx=i)
        rows.append(render_gs(pipeline, slat_mix, extrinsics, intrinsics))
        labels.append(f"SQ {i} → Style B  (A everywhere else)  [{n_vox} voxels]")

    # ---- Save outputs -----------------------------------------------------
    print("Saving outputs...")
    save_comparison_grid(rows, labels, os.path.join(output_dir, "comparison_grid.png"))
    save_sq_assignment_viz(coords, assignment, P,
                           os.path.join(output_dir, "sq_assignment.png"))

    print(f"\nDone. Results in {output_dir}/")
    print("  comparison_grid.png  — all rendering variants")
    print("  sq_assignment.png    — voxel-to-SQ assignment (3 projections)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sq-path",      default="gui/superquadrics/chair_sq.npz")
    p.add_argument("--prompt-a",     default="a wooden chair")
    p.add_argument("--prompt-b",     default="a blue metal chair")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--t0-idx",       type=int,   default=6)
    p.add_argument("--steps",        type=int,   default=12)
    p.add_argument("--cfg-strength", type=float, default=7.5)
    p.add_argument("--output-dir",   default="approach1_results")
    args = p.parse_args()

    run_experiment(
        sq_path=args.sq_path, prompt_a=args.prompt_a, prompt_b=args.prompt_b,
        seed=args.seed, t0_idx=args.t0_idx, steps=args.steps,
        cfg_strength=args.cfg_strength, output_dir=args.output_dir,
    )
