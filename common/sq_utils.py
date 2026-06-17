"""sq_utils.py — shared superquadric / voxel helpers.

These previously lived in experiments/approach1_experiment.py and were imported
from there by the benchmark and diagnostic scripts. They are factored out here so
those scripts don't depend on an experiment module. SQ-mesh normalization and the
spatial-control mesh writer live in multigen.py, next to their main consumer.
"""

from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


SQ_COLORS = np.array([
    [228,  26,  28], [55, 126, 184], [ 77, 175,  74], [152,  78, 163],
    [255, 127,   0], [166,  86,  40], [247, 129, 191], [153, 153, 153],
    [  0,   0,   0], [141, 211, 199],
], dtype=np.uint8)


def coords_to_world(coords: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
    """Integer grid coords (N, 4) → world positions in [-0.5, 0.5]³."""
    return (coords[:, 1:4].float() + 0.5) / grid_size - 0.5


def load_sq_params(npz_path: str) -> List[Dict]:
    data = np.load(npz_path)
    return [{'scale': data['scales'][k], 'shape': data['shapes'][k],
             'rotation': data['rotations'][k], 'translation': data['translations'][k]}
            for k in range(data['scales'].shape[0])]


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
    return [{
        'scale': sq['scale'],
        'shape': sq['shape'],
        'rotation': R @ np.asarray(sq['rotation'], dtype=np.float64),
        'translation': R @ np.asarray(sq['translation'], dtype=np.float64),
    } for sq in sq_params]


def compute_hard_W(voxel_pos, sq_params, mesh_center, mesh_scale):
    """One-hot voxel→SQ assignment (N, P): each voxel to the SQ it is most inside."""
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


def save_sq_assignment_viz(coords: torch.Tensor,
                           assignment: torch.Tensor,
                           n_sqs: int,
                           output_path: str,
                           panel_titles: Tuple[str, str, str] = ('Front  (XY)', 'Side   (XZ)', 'Top    (YZ)')):
    """
    Three-panel scatter plot (XY, XZ, YZ projections) of active voxels
    coloured by their assigned superquadric.

    panel_titles overrides the default Y-up labels; for TRELLIS Z-up frames
    use ('Top (XY)', 'Side (XZ)', 'Front (YZ)').
    """
    pts = coords[:, 1:4].cpu().float().numpy()   # (N, 3)
    asgn = assignment.cpu().numpy()
    colors = SQ_COLORS[asgn % len(SQ_COLORS)] / 255.0  # (N, 3) float

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # (x_idx, y_idx, x_label, y_label, title, invert_y)
    # invert_y on the XY panel flips it from a bottom-up view (matplotlib default,
    # high Y at top) to a top-down view (high Y at bottom = camera above looking down).
    projections = [
        (0, 1, 'X', 'Y', panel_titles[0], True),
        (0, 2, 'X', 'Z', panel_titles[1], False),
        (1, 2, 'Y', 'Z', panel_titles[2], False),
    ]
    for ax, (xi, yi, xl, yl, title, invert_y) in zip(axes, projections):
        ax.scatter(pts[:, xi], pts[:, yi], c=colors, s=1.5, linewidths=0)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title); ax.set_aspect('equal')
        if invert_y:
            ax.invert_yaxis()

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
