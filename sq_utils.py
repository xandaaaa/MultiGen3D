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
