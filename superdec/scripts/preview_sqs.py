"""Preview superquadrics from a finalized-dataset .npz.

Accepts either:
  - a path to a .npz file, or
  - a category+model stem (e.g. "chair_dfeb8d914d8b28ab5bb58f1e92d30bf7")

Renders the SQ mesh to a PNG (or opens an interactive window with --show).

Usage
-----
  # Render a single shape (absolute path)
  python scripts/preview_sqs.py data/dataset_20/npz/chair_dfeb8...npz

  # Render by stem (resolved under data/dataset_20/npz/)
  python scripts/preview_sqs.py chair_dfeb8d914d8b28ab5bb58f1e92d30bf7

  # Build a contact sheet of all 20 shapes
  python scripts/preview_sqs.py --all

  # Interactive open3d viewer (requires display; use --show only locally)
  python scripts/preview_sqs.py chair_dfeb8... --show
"""
from __future__ import annotations

import argparse
import sys
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
    """Same formulation as superdec.utils.predictions_handler._superquadric_mesh."""
    def f(o, m):
        return np.sign(np.sin(o)) * np.abs(np.sin(o)) ** m
    def g(o, m):
        return np.sign(np.cos(o)) * np.abs(np.cos(o)) ** m
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


def _ncolors(n: int):
    import colorsys
    return np.array([colorsys.hls_to_rgb(i / n, 0.55, 0.9) for i in range(n)])


def render_png(sq: dict, out_path: Path, title: str = "", resolution: int = 30):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    P = sq["scales"].shape[0]
    colors = _ncolors(P)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    all_verts = []
    for i in range(P):
        v, t = superquadric_mesh(
            sq["scales"][i], sq["shapes"][i], sq["rotations"][i], sq["translations"][i], resolution)
        all_verts.append(v)
        tris = v[t][..., [0, 2, 1]]  # swap y<->z for matplotlib view
        ax.add_collection3d(Poly3DCollection(tris, facecolor=colors[i], alpha=0.6, linewidth=0))
    pts = np.concatenate(all_verts)[:, [0, 2, 1]]
    mn, mx = pts.min(0), pts.max(0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 * 1.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-60)
    ax.set_title(f"{title} (P={P})", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def show_interactive(sq: dict, title: str = "", resolution: int = 30):
    import open3d as o3d
    P = sq["scales"].shape[0]
    colors = _ncolors(P)
    geoms = []
    for i in range(P):
        v, t = superquadric_mesh(
            sq["scales"][i], sq["shapes"][i], sq["rotations"][i], sq["translations"][i], resolution)
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(v)
        m.triangles = o3d.utility.Vector3iVector(t)
        m.paint_uniform_color(colors[i])
        m.compute_vertex_normals()
        geoms.append(m)
    o3d.visualization.draw_geometries(geoms, window_name=title)


def resolve(arg: str, root: Path) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    # Treat as stem
    candidate = root / "npz" / f"{arg}.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve {arg!r} (not a file, not a stem in {root/'npz'})")


def build_all_contact_sheet(root: Path, out_path: Path):
    import json
    manifest = json.loads((root / "manifest.json").read_text())
    # Render each shape, then tile 4 cols x 5 rows = 20 (grouped by category)
    tmp_dir = root / "previews_tmp"
    tmp_dir.mkdir(exist_ok=True)

    rendered = []
    for cat in sorted(manifest):
        for mid in manifest[cat]:
            npz = root / "npz" / f"{cat}_{mid}.npz"
            out = tmp_dir / f"{cat}_{mid}.png"
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
    for p in rendered:
        p.unlink()
    tmp_dir.rmdir()
    print(f"Contact sheet written to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("target", nargs="?", help="Path to .npz OR category_model stem")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"Dataset root (default: {DEFAULT_ROOT})")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path (default: next to .npz)")
    ap.add_argument("--show", action="store_true", help="Open interactive open3d viewer instead of writing PNG")
    ap.add_argument("--all", action="store_true", help="Build one 4x5 contact sheet of all shapes in --root")
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
        out = args.out or npz.with_name(npz.stem + "_preview.png")
        render_png(sq, out, title=title, resolution=args.resolution)


if __name__ == "__main__":
    main()
