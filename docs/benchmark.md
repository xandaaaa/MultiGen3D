# Benchmark

This document describes the MultiGen3D evaluation benchmark: the dataset, the
prompt files, the evaluation protocol, and how to re-run the full pipeline.

## Dataset

To evaluate MultiGen under a consistent input distribution, we built a **20-shape
superquadric benchmark** under [superdec/data/dataset_20/](../superdec/data/dataset_20/).
Each shape is a per-object `.npz`.

**Curation.** From the ShapeNet subset SuperDec was trained on, we take 10
categories (airplane, bench, cabinet, car, chair, lamp, rifle, sofa, table,
watercraft) × 8 candidates, run SuperDec to fit superquadrics, and hand-pick the
2 most distinct shapes per category. Scripts and setup notes are in
[superdec/SETUP.md](../superdec/SETUP.md).

**Editing & annotation.** SuperDec fits are imperfect, so
[superdec/scripts/sq_editor.py](../superdec/scripts/sq_editor.py) is a standalone
[viser](https://viser.studio) editor (no GPU) for hand-correcting the `.npz`.
Each shape then gets a per-SQ prompt file under
[superdec/data/dataset_20/previews/](../superdec/data/dataset_20/previews/)
(`<category>_<model_id>_annotation.txt`), with a global description plus one line
per superquadric (`SQ0: Left wing`, …).

## Prompt files

The benchmark ships two JSON files in `benchmark/`. **`prompts_augmented.json` is
the one every script should use** — `prompts.json` is the un-augmented source it
is generated from.

### `prompts.json` — the base benchmark

20 shapes, each entry describing one object and its global prompt variants:

```json
{
  "id": "airplane_d18592d9615b01bbbc0909d98a1ff2b4",
  "npz": "superdec/data/dataset_20/npz/airplane_….npz",
  "global_description": "A standard commercial airliner",
  "parts": {"SQ0": "Left wing", "SQ1": "Empennage (tail section)",
            "SQ2": "Right wing", "SQ3": "Fuselage"},
  "prompts": [
    "A commercial airliner with a white fuselage, red-tipped wings, and a blue tail section",
    "… 4 more global prompt variants …"
  ]
}
```

- `parts` maps each superquadric index to a human part label (the SuperDec
  annotation).
- `prompts` is a list of 5 global styled-description variants per shape.

This file has **no `local_prompts`**, so it only supports global-prompt scoring.

### `prompts_augmented.json` — base + per-SQ local prompts

A strict superset of `prompts.json` (identical `id` / `npz` /
`global_description` / `parts` / `prompts`) with one extra key, `local_prompts`:
for each global prompt variant, a dict mapping each SQ index to a short
per-part prompt.

```json
"local_prompts": [
  {                                       // for prompts[0]
    "0": "red-tipped metal left wing",
    "1": "blue painted metal tail section",
    "2": "red-tipped metal right wing",
    "3": "white painted metal fuselage"
  },
  { … }                                   // for prompts[1], …
]
```

These per-part prompts are what MultiGen routes to each superquadric region, and
what the CLIP per-attribute win-rate scores against. **Without them the
binding-aware metrics are empty.**

### Regenerating `prompts_augmented.json`

`benchmark/augment_prompts.py` decomposes each global prompt into one short local
prompt per superquadric with an LLM (using the `parts` labels), and writes the
augmented file:

```bash
export OPENAI_API_KEY=sk-...
python benchmark/augment_prompts.py \
    --input benchmark/prompts.json \
    --output benchmark/prompts_augmented.json
```

## Evaluation: comparative VLM ranking

We evaluate MultiGen against the geometry-matched `spacecontrol` baseline across
100 comparisons with a **comparative VLM ranking** in
[benchmark/vqa_rank.py](../benchmark/vqa_rank.py). For each `(shape, prompt)`
pair, the four rendered views (front / right / back / left) of each method are
shown side by side to a VLM, which ranks the two outputs across **five
criteria**:

- **Prompt Fidelity** — do colors/materials match the prompt's specification?
- **Structure Clarity** — does the texturing preserve recognizable part geometry?
- **Detail Quality** — are local textures clean, sharp, and artifact-free?
- **Part Assignment** — is the right appearance on the right part (no swapped/merged colors)?
- **Overall Quality** — the holistic preference.

### Results

| Method | avg_rank ↓ | overall_win ↑ |
|---|---|---|
| **MultiGen** | **1.45** | **0.59** |
| SpaceControl | 1.49 | 0.41 |

Per-criterion wins (ties not shown) tell the sharper story:

| Criterion | MultiGen wins | SpaceControl wins |
|---|---|---|
| **Prompt Fidelity** | **65** | 33 |
| **Part Assignment** | **62** | 26 |
| **Overall Quality** | **59** | 41 |
| Structure Clarity | 36 | 54 |
| Detail Quality | 24 | 71 |

MultiGen wins **decisively on the binding-aware criteria** (Prompt Fidelity, Part
Assignment) and on the **Overall** preference, exactly where global text
conditioning fails.

## Running the benchmark

We provide our renders and results in `results/`, however feel free to rerun the
benchmark as below. The full pipeline is two stages: **render**, then **score**.

### 1. Generate renders

`benchmark/run_benchmark.py` runs one approach over one or all shapes and writes
views to `<results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/view_<j>.png`.
We compare MultiGen with Spacecontrol:

```bash
python benchmark/run_benchmark.py --approach multigen     --shape-idx all
python benchmark/run_benchmark.py --approach spacecontrol --shape-idx all
```

Useful flags: `--shape-idx <n>` to render a single shape; `--steps` (default 25)
and `--force` to re-render existing views. Default prompts file is
`benchmark/prompts_augmented.json`.


### 2. VLM ranking

Set `OPENAI_API_KEY="YOUR_API_KEY"` in terminal (or pass `--api-key`) and run:

```bash
python benchmark/vqa_rank.py \
    --benchmark benchmark/prompts_augmented.json \
    --results-root results \
    --approaches multigen spacecontrol \
    --output results/vqa_ranking.json
```

Useful flags: `--shape-id <id>` to rank a single shape; `--vlm-model <name>` to
change grader (default `gpt-5-mini`). It prints a per-criterion breakdown and
writes the full per-comparison records to `--output`.

### 3. CLIP scoring (not a good comparison)

`benchmark/clip_score.py` scores the rendered views against each prompt and
reports a per-attribute win-rate between two approaches:

```bash
python benchmark/clip_score.py \
    --benchmark benchmark/prompts_augmented.json \
    --results-root results \
    --approaches multigen spacecontrol \
    --output results/clip_scores.json \
    --mask-bg
```