<h1 align="center">MultiGen: Superquadric-Aware Latent Control for 3D Object Generation </h1>

<p align="center">
  <img src="assets/teaser.png" alt="MultiGen teaser" width="100%">
</p>

**MultiGen3D** is a training-free, test-time method that gives [TRELLIS](https://github.com/microsoft/TRELLIS) **part-level appearance control**. A user authors a coarse layout as a small set of superquadric (SQ) primitives, attaches one text prompt to each part, and MultiGen generates a textured 3D asset whose appearance is *part-local* each region carries the color and material of its own prompt while geometry stays globally coherent.

## Installation

Tested on **CUDA 12.8**, NVIDIA 4090, `torch 2.8.0+cu128`.

1. Clone this repository
```sh
# Clone this repository
git clone https://github.com/xandaaaa/MultiGen.git
cd Multigen
```

2. Install the dependencies
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

## Running MultiGen

There are two ways to run MultiGen: an **interactive GUI** for authoring a single asset, and a **batch runner** for reproducing the benchmark. Both load the TRELLIS weights from `gui/` and expect a GPU. Run every command from the repository root.

### Interactive GUI

The [viser](https://viser.studio)-based editor lets you author a superquadric layout, type one prompt for any superquadric, and generate the asset with MultiGen in the loop.

```sh
# From the repo root
python gui/gui_text_image.py
# then open http://localhost:8080
```

On a remote/cluster machine, forward the viser port first:

```sh
ssh -L 8080:localhost:8080 $USER@<host>     # on your laptop
python gui/gui_text_image.py                # on the host
```

In the browser: pick a template from the dropdown (loaded from `gui/superquadrics/*_sq.npz`), edit the superquadrics, type a **Region Prompt (MultiGen)** per part, set the control slider, and click **Generate MultiGen**.

## Benchmark

MultiGen vs. the geometry-matched `spacecontrol` baseline on a **20-shape
superquadric benchmark** (100 comparisons, comparative VLM ranking):

| Method | avg_rank ↓ | overall_win ↑ |
|---|---|---|
| **MultiGen** | **1.45** | **0.59** |
| SpaceControl | 1.49 | 0.41 |

Per-criterion wins (ties not shown):

| Criterion | MultiGen wins | SpaceControl wins |
|---|---|---|
| **Prompt Fidelity** | **65** | 33 |
| **Part Assignment** | **62** | 26 |
| Structure Clarity | 36 | 54 |
| Detail Quality | 24 | 71 |
| **Overall Quality** | **59** | 41 |

MultiGen wins decisively on the binding-aware criteria (Prompt Fidelity, Part
Assignment), exactly where global text conditioning fails. Our renders and results
are provided in `results/`. The dataset, prompt files, evaluation protocol, and
commands to re-run the pipeline are in [docs/benchmark.md](docs/benchmark.md).

## Experiments

We also tried several other approaches to achieve part-aware appearance control. However the results were subpar and each approach had its own issues. We provide these experiments in this repository as well see `docs/experiments.md` for more information.

## Contributors

- **Xander Yap** — [xanyap@student.ethz.ch](mailto:xanyap@student.ethz.ch)
- **Allison Tsz Kwan Lau** — [alllau@student.ethz.ch](mailto:alllau@student.ethz.ch)
- **Zhijing Liu** — [liuzhij@student.ethz.ch](mailto:liuzhij@student.ethz.ch)

## Acknowledgements

We are grateful to our supervisors **Elisabetta Fedele**, **Sayan Deb Sarkar**, and **Ata Çelen** for their guidance and support throughout the project. We also build on [TRELLIS](https://github.com/microsoft/TRELLIS), [SpaceControl](https://github.com/spacecontrol3d/spacecontrol) and [SuperDec](https://github.com/elisabettafedele/superdec), whose work made this project possible.