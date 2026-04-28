<h1 align="center">Superquadric-Guided TRELLIS Generation</h1>

We explore how to inject **superquadric (SQ) primitive identity directly into the voxel / structured-latent (SLAT) features** of [TRELLIS](https://github.com/microsoft/TRELLIS). A user authors a coarse layout as a small set of superquadrics; the goal is to generate a textured 3D asset whose voxel latents respect that primitive structure — i.e., voxels that fall inside the same superquadric share coherent appearance and geometry features.

This codebase is built on top of TRELLIS and the [SpaceControl](https://spacecontrol3d.github.io) reference implementation. SpaceControl also conditions TRELLIS on superquadrics, but it does so by biasing the **sparse-structure** stage with a rendered control mesh. We go one step further: we modify the **SLAT denoising** stage so the voxels themselves carry superquadric information — whether by projection, reinitialization, velocity consistency, or per-primitive text conditioning. The six experiments in [experiments/](experiments/) are different answers to the question *"how should SQ identity enter the voxel latents?"*

See [commands.md](commands.md) for the directory layout, install steps, and run commands.

## Common setup: SQ → voxel mapping

All experiments share the utilities defined in [experiments/approach1_experiment.py](experiments/approach1_experiment.py):

- `load_sq_params(npz)` — loads P superquadrics (`scale`, `shape`, `rotation`, `translation`).
- `compute_mesh_normalization(sq_params)` — AABB-normalizes the SQ cloud into TRELLIS's `[-0.5, 0.5]` voxel space.
- `coords_to_world_positions(coords)` — maps sparse voxel grid coords to world positions.
- `superquadric_radial_distance(x_local, semi_axes, eps)` — standard SQ radial distance.
- A weight matrix `W ∈ R^{N×P}` that assigns each of the N sparse voxels to the P superquadrics. Approaches 1–2 use a **soft** kernel `exp(-d_r / τ)`; approaches 3–4 use a **hard** one-hot argmin over radial distance.
- `project_onto_sq_subspace(z, W)` — the key operation: a ridge-regularized least-squares projection `s* = (WᵀW)⁻¹ Wᵀ z`, with `z̄ = W s*`. Applying this to voxel latents forces voxels belonging to the same SQ to share features.

Given this shared scaffold, each experiment differs in **where** and **how** SQ structure is enforced during the TRELLIS SLAT flow.

## Experiments

### Approach 1 (original) — Constrained denoising via projection
Original formulation — `sample_slat_with_projection`.

Runs the pretrained TRELLIS SLAT flow unchanged, but **interleaves a projection onto the superquadric subspace between Euler steps**. Tunables: `blend_alpha` (0 = vanilla TRELLIS, 1 = hard project), `project_every` k steps, `project_after_frac` (only start projecting after a fraction of steps has elapsed), and an optional final hard projection. Configs compared include baseline, α ∈ {0.1, 0.3, 0.5}, α=1 on the last 50% only, and final-projection-only.

### Approach 1 (revised) — Part-level appearance transplant via hard SQ assignment
[experiments/approach1_experiment.py](experiments/approach1_experiment.py) — `hard_assign_voxels` + `transplant_features`

The revised Approach 1 takes a different route to the same question: instead of projecting voxel latents during denoising, it **tests whether SLAT features are already spatially decomposable along superquadric boundaries**. The recipe:

1. Sample one sparse structure under SpaceControl spatial control (voxel coords are shared across styles).
2. Sample two full SLATs on those coords with two different prompts — `prompt_a` (style A, e.g. "a wooden chair") and `prompt_b` (style B, e.g. "a blue metal chair").
3. **Hard-assign** every active voxel to the superquadric it is "most inside" — `argmin` of the superquadric inside-outside function `F(x)` across all P superquadrics (see `sq_inside_outside` and `hard_assign_voxels`). Unlike the soft/exponential weight matrix `W` used elsewhere, this gives a single integer SQ label per voxel.
4. For each superquadric `i`, build a mixed SLAT via `transplant_features(slat_a, slat_b, assignment, sq_idx=i)` — voxels assigned to SQ `i` carry style-B features, all other voxels keep style-A features.
5. Decode all variants (A baseline, B baseline, and every per-SQ transplant) into Gaussians, render, and compare in a single grid.

The diagnostic: if the SQ-`i` transplant shows style B localized to the spatial region of SQ `i` while the rest of the chair retains style A, the SLAT is spatially decomposable and part-level appearance editing is feasible — a strong signal that voxel latents *already* carry enough primitive-localized information for SQ-aware editing without any flow-level intervention.

### Approach 2 — P-noise initialization
[experiments/approach2_experiment.py](experiments/approach2_experiment.py) — `sample_slat_p_noise`

Instead of sampling N independent voxel noise vectors, sample only **P noise vectors** (one per superquadric) and broadcast them to voxels via `z = W s`, optionally rescaling to unit variance. The pretrained flow then denoises this low-rank initialization, with projection back onto the P-dim subspace at each step. This bakes SQ identity into the latents *by construction*, rather than projecting a posteriori.

### Approach 3 — Velocity consistency (Schemes A / B / C)
[experiments/approach3_experiment.py](experiments/approach3_experiment.py) — `sample_slat_optimized`

Uses hard W and projects the **velocity** rather than the state. At each step, the per-voxel velocity is averaged within each SQ (`v_consistent = W @ ((Wᵀ v) / counts)`) and blended back. Three schemes:
- **A — `blend_alpha`**: mix consistent vs. original velocity.
- **B — `project_after_frac`**: only enforce consistency in later denoising stages.
- **C — `rescale_noise`**: normalize the broadcast initial latent to unit variance to suppress over-saturated ("neon") color artifacts.

Compared configs: `baseline`, `hard_consist` (α=1), `blend_alpha_03` (soft), `late_consist` (α=1 on the second half).

### Approach 4 — Soft residual guidance
[experiments/approach4_experiment.py](experiments/approach4_experiment.py) — `sample_slat_refined`

A refinement of Approach 3 that addresses observed failure modes (geometry collapse, neon coloring). Keeps **independent per-voxel noise** to preserve TRELLIS's geometric detail, and only applies a soft velocity lerp `torch.lerp(v_model, v_avg, guidance_strength)` after `start_frac` of steps — enough to align color within a primitive without forcing per-voxel geometry to collapse onto the SQ subspace. Default run compares baseline vs. strength=0.15.

### Approach 5 — Local semantic guidance via spatial masks
[experiments/approach5_experiment.py](experiments/approach5_experiment.py) — `sample_slat_compositional`

Addresses **attribute bleeding**: a single global prompt like *"a wooden chair with a red seat and black legs"* often smears colors across the whole object because TRELLIS text conditioning is global. Approach 5 routes different prompts to different spatial regions by:

1. Grouping the P superquadrics into a small number of **semantic groups** via `group_sqs_by_height(sq_params, mesh_center)` — Bottom (legs), Middle (seat), Top (backrest) — using the vertical extent of the SQ cloud.
2. Collapsing the hard SQ assignment `W ∈ R^{N×P}` into a per-group mask `W_semantic ∈ R^{N×G}`.
3. At each denoising step, running the flow model **once per group** with that group's local prompt, then **spatially fusing** the per-group velocities using the group masks — each voxel's update is taken from the prompt associated with its region.

This localizes color/material conditioning to the correct spatial part without touching geometry. Default run compares a single-global-prompt baseline against the 3-prompt compositional variant (legs / seat / backrest).

### Approach 6 — Extreme composition (per-superquadric semantic routing)
[experiments/approach6_experiment.py](experiments/approach6_experiment.py) — `sample_slat_extreme`

A stress test of Approach 5: instead of G=3 semantic groups, assign **a unique text prompt to every single superquadric** (G = P). At each step the flow model is evaluated once per SQ, and the resulting velocities are fused via the hard W mask so that each voxel follows only the prompt attached to its own SQ. Missing SQ prompts fall back to a neutral default. This probes how far per-primitive conditioning can be pushed before the decoder can no longer reconcile the fragmented guidance into a coherent object — i.e., the limit of localized geometry/material generation.

## Benchmark dataset: 20 superquadric shapes

To evaluate the six approaches under a consistent input distribution, we built a small **20-shape superquadric benchmark** under [superdec/data/dataset_20/](superdec/data/dataset_20/). Each shape is a per-object `.npz` in the 4-field SuperDec format (`scales`, `shapes`, `rotations`, `translations`) consumed directly by the experiments.

### How it was curated

1. **Source.** We reuse the ShapeNet subset that [SuperDec](https://super-dec.github.io) was trained on — `dataset_small_v1.1.zip` from the Occupancy Networks AWS bucket. The full zip is 73 GB, so we stream it with `remotezip` (S3 range requests) and pull only the files we need.
2. **Categories — 10 picked.** airplane, bench, cabinet, car, chair, lamp, rifle, sofa, table, watercraft (synset IDs in [superdec/SETUP.md](superdec/SETUP.md)).
3. **Candidates — 8 per category.** Evenly-spaced indices into each category's `test.lst` give 80 candidate point clouds (~92 MB total).
4. **SuperDec decomposition.** [superdec/scripts/run_candidates.py](superdec/scripts/run_candidates.py) runs the `shapenet` checkpoint on all 80 candidates (batched, ~5 min on one GPU), saves per-object `.npz` + a 4×2 `contact_sheet.png` per category.
5. **Final pick — 2 per category.** We inspect each category's contact sheet and pick two model IDs whose shapes are visually most different. Picks are hard-coded in [superdec/scripts/finalize_dataset.py](superdec/scripts/finalize_dataset.py), which materializes the final 20 shapes into [superdec/data/dataset_20/](superdec/data/dataset_20/).

Full pipeline, environment setup, and quota notes: [superdec/SETUP.md](superdec/SETUP.md).

### Editing the superquadrics: `sq_editor.py`

SuperDec's output is often close but not perfect — a backrest may be split into two primitives, a leg may be missing, etc. [superdec/scripts/sq_editor.py](superdec/scripts/sq_editor.py) is a standalone [viser](https://viser.studio)-based web editor for hand-correcting the 20-shape dataset. No TRELLIS, no GPU — it only edits the 4-field `.npz`.

Features:
- Floating 3D labels (`SQ 0`, `SQ 1`, …) pinned to each primitive's centroid, so the color ↔ index mapping is always visible in-scene.
- Click a SQ in the 3D view to select it; its gizmo and sliders appear in the **"Selected SQ"** panel at the top of the sidebar.
- Duplicate / Delete / Add new SQ; drag the gizmo to rotate and translate; sliders for the two shape exponents and three scale axes.
- Save writes `<stem>_edited.npz` next to the original (or tick **Overwrite** to replace it).

**Running the editor:**

```sh
# 1. On your laptop — open an SSH tunnel (forward port 8080)
ssh -L 8080:localhost:8080 $USER@student-cluster.inf.ethz.ch

# 2. On the cluster (login node is fine; no GPU needed)
eval "$(/work/courses/3dv/team4/env_root/miniconda3/bin/conda shell.bash hook)"
conda activate spacecontrol
cd /work/courses/3dv/team4/MultiGen3D
python superdec/scripts/sq_editor.py

# 3. In your browser
# open http://localhost:8080
```

### Per-shape annotations

With the edited superquadrics in hand, we author per-SQ prompts in the format used by [experiments/approach6_experiment.py](experiments/approach6_experiment.py). Each shape has an annotation file under [superdec/data/dataset_20/previews/](superdec/data/dataset_20/previews/) named `<category>_<model_id>_annotation.txt`:

```
Global description: A passenger plane
SQ0: Left wing of the plane
SQ1: Empennage of the plane
SQ2: Right wing of the plane
SQ3: Fuselage of the plane
```

See [superdec/data/dataset_20/previews/airplane_d18592d9615b01bbbc0909d98a1ff2b4_annotation.txt](superdec/data/dataset_20/previews/airplane_d18592d9615b01bbbc0909d98a1ff2b4_annotation.txt) for the canonical example. The SQ indices match the labels rendered by `sq_editor.py` and `preview_sqs.py`, so what you see in the editor is what you write in the annotation.

## Benchmark: CLIP score evaluation

We evaluate each approach using **CLIP score** — cosine similarity between rendered views and a holistic text prompt that specifies per-part appearance. The benchmark is designed to measure how faithfully an approach localizes different materials and colors to the correct spatial parts of the generated object.

### Prompt suite — `benchmark/prompts.json`

[benchmark/prompts.json](benchmark/prompts.json) contains **5 appearance prompts per shape** (100 total). Each prompt is a single holistic sentence that describes the whole object while specifying distinct materials and colors at the part level, for example:

> *"A rounded chair with blue painted metal legs, red velvet seat cushion, white plastic armrests, and a dark walnut wooden backrest"*

Prompts are varied across different material palettes (wood, metal, velvet, leather, plastic, etc.) and color combinations so that no two prompts for the same shape are similar.

### Scoring script — `benchmark/clip_score.py`

[benchmark/clip_score.py](benchmark/clip_score.py) computes CLIP ViT-B/32 cosine similarity between rendered PNG images and the benchmark prompts. It averages the score across all camera views for a given (shape, prompt) pair, then reports a mean score per approach.

**Score a single renders directory against one prompt:**
```bash
python benchmark/clip_score.py \
    --renders approach1_results/renders/chair_dfeb8d914d8b28ab5bb58f1e92d30bf7/prompt_0/ \
    --prompt "A rounded chair with blue painted metal legs, red velvet seat cushion, ..."
```

**Run the full benchmark across all approaches:**
```bash
python benchmark/clip_score.py \
    --benchmark benchmark/prompts.json \
    --results-root . \
    --approaches approach1 approach2 approach5 approach6 \
    --output benchmark/results.json
```

**Expected renders layout.** Each approach's experiment script should save individual view PNGs under:
```
<approach>_results/renders/<shape_id>/prompt_<i>/view_<j>.png
```
where `shape_id` matches the `id` field in `prompts.json` and `prompt_i` indexes into the shape's 5-prompt list.
