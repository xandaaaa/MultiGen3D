"""Copy the user-picked 2-per-category shapes into a clean final dataset location.

Final layout:
  data/dataset_20/
    manifest.json         # category -> [model_id, ...]
    npz/<category>_<model_id>.npz    # experiment-format SQ params
    previews/<category>_<model_id>.png
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

SUPERDEC_DIR = Path(__file__).resolve().parent.parent
CAND_ROOT = SUPERDEC_DIR / "data" / "ShapeNet_candidates"
OUT_ROOT = SUPERDEC_DIR / "data" / "dataset_20"

# Final picks (filled in from user)
PICKS = {
    "airplane":    ("02691156", ["f44c0e1e55a3469494f3355d9c061b5a", "d18592d9615b01bbbc0909d98a1ff2b4"]),
    "bench":       ("02828884", ["f8aa82e7e4c58ce29d31c5ce17cce95d", "dd97603ce7538c15be5bbc844e6db7e"]),
    "cabinet":     ("02933112", ["bbbd4de3e7ab25ad80d6227ff9b21190", "df55c6665781293cbe53b3b9f1274310"]),
    "car":         ("02958343", ["d287b02b4679c70ab7902335d9dd94a2", "f9cad36ae25540a0bb20fd1bc4860856"]),
    "chair":       ("03001627", ["dfeb8d914d8b28ab5bb58f1e92d30bf7", "f1f670ac53799c18492d9da2668ec34c"]),
    "lamp":        ("03636649", ["dc6c499e71d04971d22730b0728b2fc9", "f228f6cd86162beb659dda512294c744"]),
    "rifle":       ("04090263", ["faa1fb485ddd6c9c8bfbe54b5d01550",  "d88c106c00384130fb5c1b0f759e2bc1"]),
    "sofa":        ("04256520", ["ed394e35b999f69edb039d8689a74349", "d9ae4cecb8203838f652f706160dc96d"]),
    "table":       ("04379243", ["f3b8c91c5dd1cb6b8722573b29f0d6d8", "d3a55d20bb9c93985a7746683ad193f0"]),
    "watercraft":  ("04530566", ["d95c49195e51912056f316a86bec8b19", "d3043fff20dad5c41dc762869682f4f"]),
}


def main():
    (OUT_ROOT / "npz").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "previews").mkdir(parents=True, exist_ok=True)

    manifest = {}
    for cat, (synset, mids) in PICKS.items():
        manifest[cat] = mids
        src_dir = CAND_ROOT / f"{synset}_{cat}"
        for mid in mids:
            src_npz = src_dir / "npz" / f"{mid}.npz"
            src_png = src_dir / "previews" / f"{mid}.png"
            if not src_npz.exists():
                raise FileNotFoundError(src_npz)
            shutil.copy2(src_npz, OUT_ROOT / "npz" / f"{cat}_{mid}.npz")
            shutil.copy2(src_png, OUT_ROOT / "previews" / f"{cat}_{mid}.png")

    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Finalized {sum(len(v) for v in manifest.values())} shapes -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
