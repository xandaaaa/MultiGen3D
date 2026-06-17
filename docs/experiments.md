# Previously attempted experiments

This document explains the previously attempted experiments that produced subpar results. The code for each experiment lives in `experiments/`.

### Approach 1 — Part-level appearance transplant

Generates SLAT_A and SLAT_B on shared coords, hard-assigns voxels to the containing SQ, and renders every per-SQ A↔B transplant in one grid.

```sh
python experiments/approach1_experiment.py \
    --sq-path gui/superquadrics/chair_sq.npz \
    --prompt-a "a wooden chair" \
    --prompt-b "a blue metal chair" \
    --steps 12
```

### Approach 2 — P-noise initialization with projection

```sh
python experiments/approach2_experiment.py \
    --sq-path gui/superquadrics/chair_sq.npz \
    --prompt "a wooden chair"
```

### Approach 3 — Velocity consistency (Schemes A/B/C)

`--exp-name` is required.

```sh
python experiments/approach3_experiment.py \
    --sq-path gui/superquadrics/chair_sq.npz \
    --prompt "a wooden chair" \
    --exp-name "my_test_v1" \
    --steps 12
```

### Approach 4 — Soft residual guidance

`--exp-name` is required.

```sh
python experiments/approach4_experiment.py \
    --sq-path gui/superquadrics/chair_sq.npz \
    --prompt "A geometric Bauhaus style chair, strictly color-blocked design, solid red seat, solid blue backrest, and bright yellow legs" \
    --exp-name "test3" \
    --steps 12
```

### Approach 5 — Semantic group routing (Top / Mid / Bottom)

No CLI args. Prompts and SQ path are hardcoded in `run_experiment()` inside [experiments/approach5_experiment.py](experiments/approach5_experiment.py) — edit `global_prompt` and `local_prompts_text` there to change the scene.

```sh
python experiments/approach5_experiment.py
```

### Approach 6 — Per-superquadric prompt routing (extreme composition)

No CLI args. Prompts are hardcoded per SQ index in `run_experiment()` inside [experiments/approach6_experiment.py](experiments/approach6_experiment.py) — edit `local_prompts_text` (keyed by SQ index) and `global_structure_prompt` to change the scene. SQ indices without an entry fall back to a neutral default.

```sh
python experiments/approach6_experiment.py
```

### Approach 7 — Coupled diffusion sampling

Builds on Approach 6 (per-SQ prompts + spatial routing) but uses *soft* voxel-to-SQ weights and adds an extra global denoising branch plus a coupling term that, at every flow step, pulls each per-SQ predicted clean sample toward the global one (squared-L2 coupling energy). The aim is to fix the seams/bleeding Approach 6 bakes in when each branch denoises with no knowledge of its neighbors. Per-SQ prompts are hardcoded in `run_experiment()` inside [experiments/approach7_experiment.py](experiments/approach7_experiment.py); `--lam` sets the coupling strength and `--tau` the softmax temperature.

```sh
python experiments/approach7_experiment.py \
    --sq-path gui/superquadrics/chair_sq.npz \
    --lam 0.3 --tau 0.02 --steps 15
```
