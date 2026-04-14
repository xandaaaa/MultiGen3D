<h1 align="center">Superquadric-Guided TRELLIS Generation</h1>

We explore how to inject **superquadric (SQ) primitive identity directly into the voxel / structured-latent (SLAT) features** of [TRELLIS](https://github.com/microsoft/TRELLIS). A user authors a coarse layout as a small set of superquadrics; the goal is to generate a textured 3D asset whose voxel latents respect that primitive structure — i.e., voxels that fall inside the same superquadric share coherent appearance and geometry features.

This codebase is built on top of TRELLIS and the [SpaceControl](https://spacecontrol3d.github.io) reference implementation. SpaceControl also conditions TRELLIS on superquadrics, but it does so by biasing the **sparse-structure** stage with a rendered control mesh. We go one step further: we modify the **SLAT denoising** stage so the voxels themselves carry superquadric information — whether by projection, reinitialization, or velocity consistency. The four experiments in [experiments/](experiments/) are different answers to the question *"how should SQ identity enter the voxel latents?"*

See [commands.txt](commands.txt) for the directory layout, install steps, and run commands.

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
