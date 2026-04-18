"""Run SuperDec on the 8-per-category candidates and render per-shape previews.

Outputs:
  data/ShapeNet_candidates/<synset>_<category>/previews/<model_id>.png
  data/ShapeNet_candidates/<synset>_<category>/npz/<model_id>.npz  (per-object, experiment format)
  data/ShapeNet_candidates/<synset>_<category>/contact_sheet.png

Run from superdec/ directory under the 'superdec' conda env, on a GPU node.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

SUPERDEC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERDEC_DIR))

from superdec.data.dataloader import SHAPENET_CATEGORIES, ShapeNet, denormalize_outdict, denormalize_points
from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def run_superdec(device: str = "cuda"):
    """Load model and the 80-candidate point clouds, return one PredictionHandler."""
    ckpt_dir = SUPERDEC_DIR / "checkpoints" / "shapenet"
    ckpt = torch.load(str(ckpt_dir / "ckpt.pt"), map_location=device, weights_only=False)
    configs = OmegaConf.load(str(ckpt_dir / "config.yaml"))

    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cands = json.loads((SUPERDEC_DIR / "data" / "ShapeNet" / "candidates.json").read_text())
    cats = cands["cats"]
    chosen = cands["chosen"]

    # Optional env-var filter: ONLY_SYNSETS="04401088,03211117,03691459"
    only = os.environ.get("ONLY_SYNSETS", "").strip()
    if only:
        keep = set(only.split(","))
        cats = {s: cats[s] for s in keep if s in cats}
        chosen = {s: chosen[s] for s in keep if s in chosen}
        print(f"filtering to synsets: {list(cats.keys())}")

    # Build a tiny ShapeNet dataset restricted to our chosen synsets.
    # We set normalize=False because the model expects points in native ShapeNet coords
    # then renormalizes internally; 'denormalize_outdict' uses translation/scale returned per-sample.
    cfg = OmegaConf.create({
        "shapenet": {
            "path": str(SUPERDEC_DIR / "data" / "ShapeNet"),
            "normalize": False,
            "categories": list(cats.keys()),
        }
    })

    # Filter to only our chosen model IDs. The ShapeNet Dataset reads <synset>/<split>.lst,
    # so write a temporary selection.lst per category and read with split='selection'.
    for synset, mids in chosen.items():
        sel = SUPERDEC_DIR / "data" / "ShapeNet" / synset / "selection.lst"
        sel.write_text("\n".join(mids) + "\n")

    dataset = ShapeNet(split="selection", cfg=cfg)
    print(f"Loaded {len(dataset)} candidate models")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=2,
        collate_fn=None,
    )

    # Build reverse map: model_id -> synset
    mid_to_synset = {mid: syn for syn, mids in chosen.items() for mid in mids}

    ph: PredictionHandler | None = None
    with torch.no_grad():
        for i, b in enumerate(loader):
            points = b["points"].to(device).float()
            b["translation"] = b["translation"].to(device)
            b["scale"] = b["scale"].to(device)
            out = model(points)
            out = denormalize_outdict(out, b["translation"], b["scale"], z_up=False)
            pts = denormalize_points(points, b["translation"], b["scale"], z_up=False)
            # model_id is a list of strings in this ShapeNet dataset
            names = [f"{mid_to_synset[m]}/{m}" for m in b["model_id"]]
            if ph is None:
                ph = PredictionHandler.from_outdict(out, pts, names)
            else:
                ph.append_outdict(out, pts, names)
            print(f"batch {i}: {len(names)} models")
    return ph


def save_per_object_npz(ph: PredictionHandler, out_root: Path):
    """Split batched PredictionHandler into per-object npz in the experiment format.

    Experiment format (see experiments/approach1_experiment.py::load_sq_params):
      scales:       [P, 3]
      shapes:       [P, 2]   (SuperDec calls this 'exponents')
      rotations:    [P, 3, 3]
      translations: [P, 3]
    Only SQs with exist > 0.5 are kept.
    """
    for i, name in enumerate(ph.names):
        synset, model_id = name.split("/")
        exist = (np.squeeze(ph.exist[i]) > 0.5)
        if not exist.any():
            print(f"  warn: {name} has no existing SQs, skipping")
            continue
        npz = {
            "scales":       ph.scale[i][exist],
            "shapes":       ph.exponents[i][exist],
            "rotations":    ph.rotation[i][exist],
            "translations": ph.translation[i][exist],
        }
        cat = SHAPENET_CATEGORIES[synset]
        out_dir = out_root / f"{synset}_{cat}" / "npz"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / f"{model_id}.npz", **npz)


def render_preview(ph: PredictionHandler, out_root: Path, resolution: int = 30):
    """Render a pointcloud + SQ mesh + overlay triptych for each candidate."""
    for i, name in enumerate(ph.names):
        synset, model_id = name.split("/")
        cat = SHAPENET_CATEGORIES[synset]
        out_dir = out_root / f"{synset}_{cat}" / "previews"
        out_dir.mkdir(parents=True, exist_ok=True)

        exist = (np.squeeze(ph.exist[i]) > 0.5)
        n_sqs = int(exist.sum())

        # Point cloud
        pc = ph.pc[i]

        # SQ mesh
        try:
            mesh = ph.get_mesh(i, resolution=resolution, colors=True)
        except Exception as e:
            print(f"  mesh fail {name}: {e}")
            mesh = None

        fig = plt.figure(figsize=(9, 3.2))
        # View: ShapeNet canonical - y up, z front (so we set elev/azim to face it)
        for ax_i, (title, data) in enumerate([
            ("pointcloud", "pc"),
            (f"SQ mesh (P={n_sqs})", "mesh"),
            ("overlay", "overlay"),
        ]):
            ax = fig.add_subplot(1, 3, ax_i + 1, projection="3d")
            ax.set_title(title, fontsize=9)
            if data in ("pc", "overlay"):
                ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c="#444", s=0.3, alpha=0.6)
            if data in ("mesh", "overlay") and mesh is not None:
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.faces)
                colors = np.asarray(mesh.visual.face_colors)[:, :3] / 255.0 if mesh.visual.face_colors is not None else None
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                triangles = verts[faces]
                # Swap to (x, z, y) to match scatter view
                triangles = triangles[..., [0, 2, 1]]
                pc_kwargs = dict(alpha=0.45, linewidth=0)
                if colors is not None:
                    pc_kwargs["facecolors"] = colors
                coll = Poly3DCollection(triangles, **pc_kwargs)
                ax.add_collection3d(coll)
            _axes_equal(ax, pc[:, [0, 2, 1]])
            ax.set_axis_off()
            ax.view_init(elev=20, azim=-60)
        fig.suptitle(f"{cat} / {model_id[:12]}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model_id}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("Rendered previews")


def _axes_equal(ax, pts):
    mn, mx = pts.min(0), pts.max(0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 * 1.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def build_contact_sheets(out_root: Path, only_synsets: set[str] | None = None):
    """Tile the 8 per-shape previews into a 4x2 grid per category."""
    for cat_dir in sorted(out_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        if only_synsets is not None:
            synset = cat_dir.name.split("_", 1)[0]
            if synset not in only_synsets:
                continue
        prev_dir = cat_dir / "previews"
        pngs = sorted(prev_dir.glob("*.png"))
        if not pngs:
            continue
        imgs = [Image.open(p) for p in pngs]
        w, h = imgs[0].size
        cols, rows = 2, (len(imgs) + 1) // 2
        sheet = Image.new("RGB", (cols * w, rows * h), "white")
        for idx, im in enumerate(imgs):
            r, c = idx // cols, idx % cols
            sheet.paste(im, (c * w, r * h))
        sheet.save(cat_dir / "contact_sheet.png")
        print(f"  contact sheet: {cat_dir.name} ({len(imgs)} shapes)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    out_root = SUPERDEC_DIR / "data" / "ShapeNet_candidates"
    out_root.mkdir(exist_ok=True)

    # When filtering, use a synset-specific cache name
    only = os.environ.get("ONLY_SYNSETS", "").strip()
    raw_name = "_raw_batched.npz" if not only else f"_raw_batched_{only.replace(',', '_')}.npz"
    raw_path = out_root / raw_name
    if raw_path.exists() and os.environ.get("RERUN_SUPERDEC") != "1":
        print(f"reusing cached predictions at {raw_path}")
        ph = PredictionHandler.from_npz(str(raw_path))
    else:
        ph = run_superdec(device=device)
        ph.save_npz(str(raw_path))
        print("saved raw batched npz")

    save_per_object_npz(ph, out_root)
    render_preview(ph, out_root)
    only_set = set(only.split(",")) if only else None
    build_contact_sheets(out_root, only_synsets=only_set)

    # Cleanup the selection.lst tmp files
    for synset in json.loads((SUPERDEC_DIR / "data" / "ShapeNet" / "candidates.json").read_text())["cats"]:
        sel = SUPERDEC_DIR / "data" / "ShapeNet" / synset / "selection.lst"
        if sel.exists():
            sel.unlink()

    print(f"\nDone. Review contact sheets in {out_root}/<synset>_<category>/contact_sheet.png")


if __name__ == "__main__":
    main()
