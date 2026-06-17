<h1 align="center">MultiGen: Superquadric-Aware Latent Control for 3D Object Generation </h1>

**MultiGen** is a training-free, test-time method that gives [TRELLIS](https://github.com/microsoft/TRELLIS) **part-level appearance control**. A user authors a coarse layout as a small set of superquadric (SQ) primitives, attaches one text prompt to each part, and MultiGen generates a textured 3D asset whose appearance is *part-local* — each region carries the color and material of its own prompt — while geometry stays globally coherent.

TRELLIS conditions every voxel on one global prompt, so compositional descriptions smear attributes across the whole object. MultiGen moves control into the SLAT denoising stage, where this attribute binding is decided.

## Installation

Tested on **CUDA 12.8**, NVIDIA 4090, `torch 2.8.0+cu128`.

```sh
# Check your CUDA toolkit
nvcc --version

# Create the environment
conda create -n multigen python=3.10 -y
conda activate multigen

# PyTorch (see https://pytorch.org/get-started/locally/ for your setup)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Core Python dependencies
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph \
    transformers psutil viser tensorboard pandas lpips
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Attention + sparse kernels
pip install xformers==0.0.32.post1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
pip install spconv-cu120

# Rendering extensions
mkdir -p /tmp/extensions

git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast --no-build-isolation

git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
pip install /tmp/extensions/diffoctreerast --no-build-isolation

git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

cp -r extensions/vox2seq /tmp/extensions/vox2seq
pip install /tmp/extensions/vox2seq --no-build-isolation
```

Sanity check the CUDA + sparse-conv install:

```sh
python -c "import torch; print(torch.cuda.is_available()); import spconv; print('spconv OK')"
```

## Method

MultiGen runs the pretrained TRELLIS SLAT flow unchanged, but replaces the global guidance signal with **compositional classifier-free guidance** routed through superquadric masks (implementation: [multigen.py](multigen.py), `sample_multigen_slat` / `multigen_generate`).

Each active voxel is assigned to the superquadric it is "most inside" (argmin radial distance), giving a per-prompt voxel mask. All regions then share **one noise tensor and one sampling trajectory** — no slicing or seam stitching. At each denoising step the shared latent is denoised once per unique region prompt plus one shared negative pass (`#unique prompts + 1` passes), and the per-region CFG velocities are blended in voxel space through the masks, so each voxel follows only its own prompt while geometry stays globally coherent. `spacecontrol` (same geometry, single global prompt, no routing) is the apples-to-apples reference.

## Running MultiGen

There are two ways to run MultiGen: an **interactive GUI** for authoring a single asset, and a **batch runner** for reproducing the benchmark. Both load the TRELLIS weights from `gui/` and expect a GPU. Run every command from the repository root (the scripts use paths relative to it).

### Interactive GUI

The [viser](https://viser.studio)-based editor lets you author a superquadric layout, type one prompt per region, and generate the asset with MultiGen in the loop.

```sh
# From the repo root (the GUI loads weights via from_pretrained("gui")
# and reads templates from gui/superquadrics/, both relative paths)
python gui/gui_text_image.py
# then open http://localhost:8080
```

On a remote/cluster machine, forward the viser port first:

```sh
ssh -L 8080:localhost:8080 $USER@<host>     # on your laptop
python gui/gui_text_image.py                # on the host (needs a GPU)
```

In the browser: pick a template from the dropdown (loaded from `gui/superquadrics/*_sq.npz`), edit the superquadrics, type a **Region Prompt (MultiGen)** per part, set the control slider, and click **Generate MultiGen**.

### Programmatic

`multigen.py` exposes the method directly — `sample_multigen_slat(...)` returns the SLAT and `multigen_generate(...)` returns a decoded `(gaussian, mesh)`. Both take a TRELLIS pipeline, the sparse `coords`, a `{sq_idx: cond}` map of per-region text conditionings, the global conditioning, and the SQ params with their `(mesh_center, mesh_scale)` normalization (see [benchmark/run_benchmark.py](benchmark/run_benchmark.py) `run_multigen` for a complete call site).

## Benchmark dataset: 20 superquadric shapes

To evaluate MultiGen under a consistent input distribution, we built a **20-shape superquadric benchmark** under [superdec/data/dataset_20/](superdec/data/dataset_20/). Each shape is a per-object `.npz` in the 4-field SuperDec format (`scales`, `shapes`, `rotations`, `translations`) consumed directly by the generation pipeline.

**Curation.** From the ShapeNet subset SuperDec was trained on, we take 10 categories (airplane, bench, cabinet, car, chair, lamp, rifle, sofa, table, watercraft) × 8 candidates, run SuperDec to fit superquadrics, and hand-pick the 2 most distinct shapes per category. Scripts and setup notes are in [superdec/SETUP.md](superdec/SETUP.md).

**Editing & annotation.** SuperDec fits are imperfect, so [superdec/scripts/sq_editor.py](superdec/scripts/sq_editor.py) is a standalone [viser](https://viser.studio) editor (no GPU) for hand-correcting the `.npz`. Each shape then gets a per-SQ prompt file under [superdec/data/dataset_20/previews/](superdec/data/dataset_20/previews/) (`<category>_<model_id>_annotation.txt`), with a global description plus one line per superquadric (`SQ0: Left wing`, …) — the SQ indices match the editor labels.

## Evaluation: comparative VLM ranking

We evaluate MultiGen against the geometry-matched `spacecontrol` baseline with a **comparative VLM ranking** in [benchmark/vqa_rank.py](benchmark/vqa_rank.py). For each `(shape, prompt)` pair, the four rendered views (front / right / back / left) of each method are shown side by side to a VLM, which ranks the two outputs across **five criteria** adapted from the SuperDec protocol:

- **Prompt Fidelity** — do colors/materials match the prompt's specification?
- **Structure Clarity** — does the texturing preserve recognizable part geometry?
- **Detail Quality** — are local textures clean, sharp, and artifact-free?
- **Part Assignment** — is the right appearance on the right part (no swapped/merged colors)?
- **Overall Quality** — the holistic preference.

Why a *comparative* VLM ranking rather than global CLIP: our prompts bind a distinct material/color to each part, and global CLIP collapses image and text into single vectors — it scores a render with the right colors present *anywhere* as well as one with the colors on the *correct* parts, so it is blind to exactly the attribute binding we care about. Showing both methods to a VLM and asking it to rank them directly targets per-part correctness.

### Results

Across the 100 `(shape, prompt)` comparisons (`gpt-5-mini` grader, recorded in [results/vqa_ranking.json](results/vqa_ranking.json)):

| Method | avg_rank ↓ | win_rate ↑ | overall_win ↑ |
|---|---|---|---|
| **MultiGen** | **1.45** | **0.55** | **0.59** |
| SpaceControl | 1.49 | 0.51 | 0.41 |

Per-criterion wins (ties not shown) tell the sharper story:

| Criterion | MultiGen wins | SpaceControl wins |
|---|---|---|
| **Prompt Fidelity** | **65** | 33 |
| **Part Assignment** | **62** | 26 |
| **Overall Quality** | **59** | 41 |
| Structure Clarity | 36 | 54 |
| Detail Quality | 24 | 71 |

MultiGen wins **decisively on the binding-aware criteria** (Prompt Fidelity, Part Assignment) and on the **Overall** preference, exactly where global text conditioning fails. SpaceControl edges ahead on Structure Clarity and Detail Quality — it texturizes a single global prompt very cleanly — but at the cost of putting attributes on the wrong parts. This trade is the central claim: for part-aware 3D texturing, *where* guidance is applied in the latent matters more than how cleanly a single global prompt is rendered.

### Running the benchmark

The full pipeline is two stages — **render**, then **score**. Run everything from the repository root.

**1. Generate renders.** `benchmark/run_benchmark.py` runs one approach over one or all shapes and writes views to `<results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/view_<j>.png`. The three approaches are `baseline` (text-driven structure, global prompt), `spacecontrol` (SQ-controlled structure, single global prompt — the geometry-matched reference), and `multigen` (SQ-controlled structure + per-region compositional CFG):

```bash
python benchmark/run_benchmark.py --approach multigen     --shape-idx all
python benchmark/run_benchmark.py --approach spacecontrol --shape-idx all
python benchmark/run_benchmark.py --approach baseline     --shape-idx all   # optional
```

Useful flags: `--shape-idx <n>` to render a single shape; `--steps` (default 15); `--force` to re-render existing views; `--resolution 256` if 16 GB GPUs OOM; multigen-only `--local-cfg` (per-region strength, default 15.0), `--soft-tau` (soft masks; omit for hard one-hot), and `--t0-idx` (spatial-control strength). Default prompts file is `benchmark/prompts_augmented.json`.

**2a. VLM ranking (primary metric).** `benchmark/vqa_rank.py` shows both methods' four views to a VLM and ranks them across the five criteria. Set `OPENAI_API_KEY` (or pass `--api-key`):

```bash
python benchmark/vqa_rank.py \
    --benchmark benchmark/prompts_augmented.json \
    --results-root results \
    --approaches multigen spacecontrol \
    --output results/vqa_ranking.json
```

Useful flags: `--shape-id <id>` to rank a single shape; `--vlm-model <name>` to change grader (default `gpt-5-mini`). It prints a per-criterion breakdown and writes the full per-comparison records to `--output`.

**2b. CLIP win-rate (optional, secondary).** `benchmark/clip_score.py` reports a per-attribute CLIP win-rate. Pass `--benchmark` (the augmented prompts carry the per-part `local_prompts` the win-rate needs) for the full pairwise comparison:

```bash
python benchmark/clip_score.py \
    --benchmark benchmark/prompts_augmented.json \
    --results-root results \
    --approaches multigen spacecontrol \
    --clip-model ViT-L/14 --mask-bg \
    --output results/clip_scores.json
```

To score a single renders directory against one prompt instead, pass `--renders <dir> --prompt "..."`. The `--clip-model ViT-L/14` and `--mask-bg` flags give the strongest attribute signal.

**Diagnostics (optional).** `benchmark/gen_sq_assignment.py` dumps the per-shape voxel→superquadric assignment visualization used to sanity-check the routing masks:

```bash
python benchmark/gen_sq_assignment.py --out-dir results/sq_diagnostics
```
