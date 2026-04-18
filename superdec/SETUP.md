# SuperDec benchmark-dataset setup

This file documents how the 20-shape superquadric benchmark dataset under
[data/dataset_20/](data/dataset_20) is produced, so the pipeline can be
re-run or extended to more shapes later.

## What's here

We use [SuperDec](https://super-dec.github.io) to decompose ShapeNet objects
into superquadric primitives, then filter the results down to **20 hand-picked
shapes (2 per category × 10 categories)**. The resulting `.npz` files match
the format consumed by the `load_sq_params` helper in
[../experiments/approach1_experiment.py](../experiments/approach1_experiment.py):

```
scales:       (P, 3)      float
shapes:       (P, 2)      float  # SuperDec calls these 'exponents'
rotations:    (P, 3, 3)   float
translations: (P, 3)      float
```

The 10 categories (with ShapeNet synset IDs):

| synset | name |
|:---|:---|
| `02691156` | airplane |
| `02828884` | bench |
| `02933112` | cabinet |
| `02958343` | car |
| `03001627` | chair |
| `03636649` | lamp |
| `04090263` | rifle |
| `04256520` | sofa |
| `04379243` | table |
| `04530566` | watercraft |

## Environment setup

SuperDec has its own conda env (separate from `spacecontrol` so TRELLIS and
SuperDec don't fight over torch/CUDA versions).

```sh
eval "$(/work/courses/3dv/team4/env_root/miniconda3/bin/conda shell.bash hook)"

# Create env at the shared location under env_root (NOT ~/.conda/envs)
conda create -p /work/courses/3dv/team4/env_root/miniconda3/envs/superdec \
    -c conda-forge --override-channels python=3.11 -y
conda activate superdec

# Install deps (use /work for pip cache so the 100 GB ceph quota on the team dir
# doesn't fill up via ~/.cache/pip)
cd /work/courses/3dv/team4/MultiGen3D/superdec
pip install --cache-dir /work/courses/3dv/team4/cache_alllau/pip -r requirements.txt
pip install --cache-dir /work/courses/3dv/team4/cache_alllau/pip -e .
```

> Notes
> - The team work dir has a **100 GB ceph quota** (`ceph.quota.max_bytes`).
>   Always redirect pip/hf/torch caches to `/work/courses/3dv/team4/cache_alllau/`
>   to avoid filling your home quota.
> - On first run inside a GPU node, `superdec/functional/backend.py` JIT-compiles
>   a CUDA extension (`_pvcnn_backend`). This takes ~1 min. The login node has
>   no GPU, so any `import superdec.superdec` there will fail — that's expected.

## Checkpoints

```sh
cd /work/courses/3dv/team4/MultiGen3D/superdec
bash scripts/download_checkpoints.sh
```

This writes to `checkpoints/shapenet/` and `checkpoints/normalized/`. We use
the **`shapenet`** checkpoint (trained on native-scale ShapeNet); the
`normalized` one is for objects pulled from real scenes and unused here.

## Dataset pipeline

The official SuperDec flow is to download all 73 GB of
`dataset_small_v1.1.zip` from `s3.eu-central-1.amazonaws.com` and extract
`*pointcloud.npz` and `*.lst` via `scripts/download_shapenet.sh`. We can't do
that here — it blows the 100 GB team quota. Instead we read the zip remotely
via `remotezip` (S3 supports HTTP range requests), pull only the files we
need, and pick the final shapes through visual selection.

### Step 1 — fetch `test.lst` files + 8 candidate point clouds per category

Already done. Re-run with:

```sh
conda activate superdec
python - <<'EOF'
from remotezip import RemoteZip
import os, shutil, json
url = "https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip"
cats = {
    "03001627": "chair", "04379243": "table",  "04256520": "sofa",
    "03636649": "lamp",  "02933112": "cabinet","02828884": "bench",
    "02691156": "airplane","02958343": "car", "04530566": "watercraft",
    "04090263": "rifle",
}
out_root = "data/ShapeNet"
tmp_dir = "data/.zip_tmp"; os.makedirs(tmp_dir, exist_ok=True)
chosen = {}
with RemoteZip(url) as zf:
    for synset in cats:
        for split in ("test.lst", "train.lst", "val.lst"):
            try: zf.extract(f"ShapeNet/{synset}/{split}", path=tmp_dir)
            except KeyError: pass
        src = f"{tmp_dir}/ShapeNet/{synset}"
        dst = f"{out_root}/{synset}"; os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src): shutil.move(os.path.join(src,f), os.path.join(dst,f))
        with open(os.path.join(dst, "test.lst")) as fh:
            models = [l.strip() for l in fh if l.strip()]
        # 8 evenly-spaced picks from test.lst for diversity
        idxs = [int(round(i * len(models) / 8)) for i in range(8)]
        chosen[synset] = [models[i] for i in idxs]
        for mid in chosen[synset]:
            d = f"{out_root}/{synset}/{mid}/pointcloud.npz"
            if os.path.exists(d): continue
            zf.extract(f"ShapeNet/{synset}/{mid}/pointcloud.npz", path=tmp_dir)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.move(f"{tmp_dir}/ShapeNet/{synset}/{mid}/pointcloud.npz", d)
        print(synset, cats[synset], "OK")
json.dump({"cats": cats, "chosen": chosen},
          open(f"{out_root}/candidates.json","w"), indent=2)
shutil.rmtree(tmp_dir, ignore_errors=True)
EOF
```

Result: 80 point clouds (8 per category), **~92 MB total**, plus all
`test.lst` / `train.lst` / `val.lst` files, under `data/ShapeNet/<synset>/`.
Record of what got picked is in `data/ShapeNet/candidates.json`.

### Step 2 — decompose the 80 candidates with SuperDec

```sh
sbatch scripts/run_candidates.sbatch   # ~5 min on studgpu-node01
```

This calls [scripts/run_candidates.py](scripts/run_candidates.py), which:

1. Loads the `shapenet` checkpoint and runs SuperDec in eval mode on all
   candidates (batched).
2. Saves the raw batched prediction at
   `data/ShapeNet_candidates/_raw_batched.npz` (used as a cache — re-running
   the sbatch won't re-run the GPU inference unless you set
   `RERUN_SUPERDEC=1`).
3. Splits the batched prediction into **per-object** `.npz` files in the
   format consumed by the experiments:

   ```
   data/ShapeNet_candidates/<synset>_<category>/
     npz/<model_id>.npz
     previews/<model_id>.png         # pointcloud | SQ mesh | overlay
     contact_sheet.png                # 4x2 grid of the 8 candidates
   ```

To run only a subset of synsets, set `ONLY_SYNSETS` in the sbatch (comma-
separated), e.g. `ONLY_SYNSETS="03001627,04379243"`. This is how the
3-category `scripts/run_candidates_3new.sbatch` was produced for the
telephone/display/loudspeaker review (kept in-repo for reference; those
categories were dropped).

### Step 3 — pick 2 per category from the contact sheets

You open each `data/ShapeNet_candidates/<synset>_<category>/contact_sheet.png`
and pick two model IDs whose shapes look most different. The picks get hard-
coded into [scripts/finalize_dataset.py](scripts/finalize_dataset.py) under
`PICKS`.

### Step 4 — materialize the final dataset

```sh
python scripts/finalize_dataset.py
```

Copies the 20 selected `.npz` + `.png` files into
[data/dataset_20/](data/dataset_20):

```
data/dataset_20/
├── manifest.json                            # category -> [model_id, model_id]
├── npz/
│   ├── airplane_<id>.npz
│   ├── chair_<id>.npz
│   └── ... (20 total)
├── previews/
│   ├── airplane_<id>.png
│   └── ... (20 total; each is the pointcloud|SQ-mesh|overlay triptych)
└── contact_sheet.png                        # 4x5 grid of all 20 SQ meshes
```

`contact_sheet.png` at the top level is produced by
`scripts/preview_sqs.py --all` (see below).

## Previewing superquadrics

[scripts/preview_sqs.py](scripts/preview_sqs.py) renders SQ meshes directly
from the final `.npz` format — no SuperDec model needed.

```sh
# Single shape, by category_modelID stem
python scripts/preview_sqs.py chair_dfeb8d914d8b28ab5bb58f1e92d30bf7

# Single shape, by explicit path + custom output
python scripts/preview_sqs.py data/dataset_20/npz/chair_dfeb8...npz --out /tmp/foo.png

# Build one 4x5 contact sheet of all 20 shapes
python scripts/preview_sqs.py --all

# Interactive open3d window (only on a machine with a display)
python scripts/preview_sqs.py chair_dfeb8... --show
```

The same script also works on `.npz` files produced anywhere else by the
`finalize_dataset.py` pipeline — just pass a path or `--root` pointing to a
different `dataset_*` directory.

## File map

```
superdec/
├── SETUP.md                         # this file
├── README.md                        # upstream SuperDec README
├── checkpoints/                     # shapenet + normalized ckpts
├── data/
│   ├── ShapeNet/                    # only `.lst` + 80 raw pointclouds (~92 MB)
│   │   └── candidates.json          # synset -> 8 picked model_ids
│   ├── ShapeNet_candidates/         # 8-per-category SuperDec outputs + contact sheets
│   └── dataset_20/                  # 20-shape final benchmark dataset
│       ├── manifest.json
│       ├── npz/
│       ├── previews/
│       └── contact_sheet.png
├── scripts/
│   ├── download_checkpoints.sh      # upstream
│   ├── download_shapenet.sh         # upstream (NOT USED — too big for our quota)
│   ├── run_on_shapenet.sh           # upstream
│   ├── run_on_scene.sh              # upstream
│   ├── run_candidates.py            # our: SuperDec on 8-per-cat candidates + render
│   ├── run_candidates.sbatch        # our: SLURM wrapper for run_candidates.py
│   ├── run_candidates_3new.sbatch   # our: filtered variant (reference only)
│   ├── finalize_dataset.py          # our: copy PICKS into dataset_20/
│   └── preview_sqs.py               # our: render any dataset_20/npz/*.npz
├── superdec/                        # upstream package
├── configs/                         # upstream hydra configs
└── requirements.txt
```

## Adding more shapes later

To extend beyond 20 shapes you have two options, depending on whether the
addition is *more categories* or *more shapes per category*:

1. **New categories.** Append synsets to `cats` and run Step 1 again to pull
   their `.lst` + 8 pointclouds. Submit `run_candidates.py` with
   `ONLY_SYNSETS="<new_synset_list>"`. Review their contact sheets, then
   extend `PICKS` in `finalize_dataset.py` and rerun it.

2. **More candidates per existing category.** Change `N_CANDIDATES` in the
   Step 1 snippet (or modify `scripts/run_candidates.py` to accept a CLI
   override) and re-run. Delete the cached `_raw_batched.npz` first, or set
   `RERUN_SUPERDEC=1` in the sbatch, so SuperDec actually re-runs on the
   enlarged candidate set.
