"""Preview superquadrics from a finalized-dataset .npz.

Renders each SQ with a distinct color and overlays its numeric index at the
centroid, so you can author per-SQ prompts in the approach6_experiment.py
style (e.g. 'SQ 3: a green plastic chair backrest').

Accepts either:
  - a path to a .npz file, or
  - a category+model stem (e.g. "chair_dfeb8d914d8b28ab5bb58f1e92d30bf7")

Usage
-----
  # Render a single shape (4-view layout + legend)
  python scripts/preview_sqs.py chair_dfeb8d914d8b28ab5bb58f1e92d30bf7

  # Render all 20 shapes in dataset_20/ and rebuild the all-shapes contact sheet
  python scripts/preview_sqs.py --all

  # Interactive open3d window (requires display)
  python scripts/preview_sqs.py chair_dfeb8... --show
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

SUPERDEC_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = SUPERDEC_DIR / "data" / "dataset_20"


def load_sq(npz_path: Path) -> dict:
    d = np.load(npz_path)
    return {
        "scales":       d["scales"],
        "shapes":       d["shapes"],
        "rotations":    d["rotations"],
        "translations": d["translations"],
    }


def superquadric_mesh(scale, exponents, rotation, translation, N: int = 30):
    """Tessellate one SQ into (verts, faces). Matches superdec.utils.predictions_handler."""
    def f(o, m): return np.sign(np.sin(o)) * np.abs(np.sin(o)) ** m
    def g(o, m): return np.sign(np.cos(o)) * np.abs(np.cos(o)) ** m
    u = np.linspace(-np.pi, np.pi, N, endpoint=True)
    v = np.linspace(-np.pi / 2.0, np.pi / 2.0, N, endpoint=True)
    u = np.tile(u, N); v = np.repeat(v, N)
    if np.linalg.det(rotation) < 0:
        u = u[::-1]
    x = scale[0] * g(v, exponents[0]) * g(u, exponents[1])
    y = scale[1] * g(v, exponents[0]) * f(u, exponents[1])
    z = scale[2] * f(v, exponents[0])
    x[:N] = 0.0
    x[-N:] = 0.0
    verts = np.stack([x, y, z], axis=1)
    verts = (rotation @ verts.T).T + translation
    tris = []
    for i in range(N - 1):
        for j in range(N - 1):
            tris.append([i * N + j, i * N + j + 1, (i + 1) * N + j])
            tris.append([(i + 1) * N + j, i * N + j + 1, (i + 1) * N + (j + 1)])
    for i in range(N - 1):
        tris.append([i * N + (N - 1), i * N, (i + 1) * N + (N - 1)])
        tris.append([(i + 1) * N + (N - 1), i * N, (i + 1) * N])
    tris.append([(N - 1) * N + (N - 1), (N - 1) * N, (N - 1)])
    tris.append([(N - 1), (N - 1) * N, 0])
    return verts, np.array(tris)


def sq_color(idx: int, total: int) -> tuple[float, float, float]:
    """Deterministic distinct color per SQ index. Uses golden-ratio hue stepping
    so small P shapes (4 SQs) and large P shapes (16 SQs) both look well-separated."""
    # Golden ratio hue stepping -> good separation without relying on total P
    golden = 0.61803398875
    h = (idx * golden) % 1.0
    return colorsys.hls_to_rgb(h, 0.55, 0.9)


# --- Rendering -----------------------------------------------------------------

VIEWS = [
    ("front",     20, -90),   # elev, azim
    ("side",      20,   0),
    ("top",       85, -90),
    ("isometric", 25, -60),
]


def render_png(sq: dict, out_path: Path, title: str = "", resolution: int = 30):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    P = sq["scales"].shape[0]
    colors = [sq_color(i, P) for i in range(P)]

    # Precompute meshes once per SQ
    meshes = []
    for i in range(P):
        v, t = superquadric_mesh(
            sq["scales"][i], sq["shapes"][i], sq["rotations"][i], sq["translations"][i], resolution)
        meshes.append((v, t))

    # Global bounds across all SQs
    all_verts = np.concatenate([m[0] for m in meshes])
    pts_view = all_verts[:, [0, 2, 1]]  # swap y<->z so matplotlib's z-up looks like ShapeNet y-up
    mn, mx = pts_view.min(0), pts_view.max(0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 * 1.1

    # 2x2 views + a legend column
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55], wspace=0.02, hspace=0.05)

    for ax_i, (vname, elev, azim) in enumerate(VIEWS):
        r_grid = ax_i // 2
        c_grid = ax_i % 2
        ax = fig.add_subplot(gs[r_grid, c_grid], projection="3d")
        for i, (v, t) in enumerate(meshes):
            tris = v[t][..., [0, 2, 1]]
            ax.add_collection3d(Poly3DCollection(tris, facecolor=colors[i], alpha=0.75, linewidth=0))
        # Label at each SQ centroid (project using the same view)
        for i, (v, _) in enumerate(meshes):
            centroid = v.mean(0)[[0, 2, 1]]
            ax.text(centroid[0], centroid[1], centroid[2], str(i),
                    color="black", fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black",
                              boxstyle="round,pad=0.15", linewidth=0.5), zorder=10)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(vname, fontsize=9)

    # Legend on the right
    lgax = fig.add_subplot(gs[:, 2])
    lgax.set_xlim(0, 1); lgax.set_ylim(0, 1)
    lgax.set_axis_off()
    lgax.set_title("SQ index", fontsize=10, loc="left")
    n_rows = P
    y_step = 1.0 / max(n_rows, 12)
    for i in range(P):
        y = 1 - (i + 0.5) * y_step
        lgax.add_patch(plt.Rectangle((0.0, y - y_step * 0.35), 0.15, y_step * 0.7,
                                     facecolor=colors[i], edgecolor="black", linewidth=0.4))
        lgax.text(0.20, y, f"SQ {i}", fontsize=9, va="center")

    fig.suptitle(f"{title}  (P={P})", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def show_interactive(sq: dict, title: str = "", resolution: int = 30):
    import open3d as o3d
    P = sq["scales"].shape[0]
    geoms = []
    for i in range(P):
        v, t = superquadric_mesh(
            sq["scales"][i], sq["shapes"][i], sq["rotations"][i], sq["translations"][i], resolution)
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(v)
        m.triangles = o3d.utility.Vector3iVector(t)
        m.paint_uniform_color(sq_color(i, P))
        m.compute_vertex_normals()
        geoms.append(m)
    o3d.visualization.draw_geometries(geoms, window_name=title)


def resolve(arg: str, root: Path) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    candidate = root / "npz" / f"{arg}.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve {arg!r} (not a file, not a stem in {root/'npz'})")


def build_all_contact_sheet(root: Path, out_path: Path):
    import json
    manifest = json.loads((root / "manifest.json").read_text())
    # Re-render each shape's labelled preview into previews/
    prev_dir = root / "previews"
    prev_dir.mkdir(exist_ok=True)

    rendered = []
    for cat in sorted(manifest):
        for mid in manifest[cat]:
            npz = root / "npz" / f"{cat}_{mid}.npz"
            out = prev_dir / f"{cat}_{mid}.png"
            render_png(load_sq(npz), out, title=f"{cat} / {mid[:10]}")
            rendered.append(out)

    imgs = [Image.open(p) for p in rendered]
    w, h = imgs[0].size
    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for idx, im in enumerate(imgs):
        r, c = idx // cols, idx % cols
        sheet.paste(im, (c * w, r * h))
    sheet.save(out_path)
    print(f"Contact sheet written to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("target", nargs="?", help="Path to .npz OR category_model stem")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Dataset root (default: {DEFAULT_ROOT})")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path")
    ap.add_argument("--show", action="store_true", help="Open interactive open3d viewer")
    ap.add_argument("--all", action="store_true", help="Re-render all dataset shapes + rebuild contact sheet")
    ap.add_argument("--resolution", type=int, default=30)
    args = ap.parse_args()

    if args.all:
        out = args.out or (args.root / "contact_sheet.png")
        build_all_contact_sheet(args.root, out)
        return

    if not args.target:
        ap.error("Either pass a target (.npz path or stem) or use --all.")

    npz = resolve(args.target, args.root)
    sq = load_sq(npz)
    title = npz.stem

    if args.show:
        show_interactive(sq, title=title, resolution=args.resolution)
    else:
        out = args.out or npz.with_name(npz.stem + "_labelled.png")
        render_png(sq, out, title=title, resolution=args.resolution)


if __name__ == "__main__":
    main()
