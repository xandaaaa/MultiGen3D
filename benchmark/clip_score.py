"""
CLIP scoring for MultiGen3D benchmark.

Usage — score a single directory of renders against a prompt:
    python benchmark/clip_score.py \
        --renders approach1_results/renders/chair_dfeb8d914d8b28ab5bb58f1e92d30bf7/prompt_0/ \
        --prompt "A rounded chair with blue painted metal legs, red velvet seat cushion, ..."

Usage — score all approaches against the full benchmark:
    python benchmark/clip_score.py \
        --benchmark benchmark/prompts_augmented.json \
        --results-root results \
        --approaches multigen spacecontrol

Expected renders directory layout (created by the experiment scripts):
    <results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/view_<j>.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from PIL import Image


# Short aliases (openai-clip naming) -> HuggingFace model ids.
CLIP_MODEL_ALIASES = {
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-L/14": "openai/clip-vit-large-patch14",
    "ViT-L/14@336px": "openai/clip-vit-large-patch14-336",
}


def load_clip(model_name: str = "ViT-B/32"):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        sys.exit("transformers not installed. Run: pip install transformers")
    hf_id = CLIP_MODEL_ALIASES.get(model_name, model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(hf_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(hf_id)
    return model, processor, device


def neutralize_background(img: Image.Image, grey: int = 128, tol: int = 12) -> Image.Image:
    """Replace a near-uniform pure-black or pure-white background with neutral grey.

    The benchmark renderer paints the background black when the prompt mentions
    "white" and white otherwise (run_benchmark._render_gs). Both are strong colour
    cues that bias CLIP, and they differ per-prompt so raw scores aren't comparable.
    We can't re-render (compute-limited), so we composite the background to a fixed
    neutral grey at scoring time. Only pixels matching the detected corner colour
    (within tol) are replaced, so the object itself is untouched.
    """
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    corners = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    bg = corners.mean(0)
    # Only act on near-pure black/white backgrounds; leave anything else alone.
    if not (np.all(bg <= tol) or np.all(bg >= 255 - tol)):
        return img.convert("RGB")
    mask = np.all(np.abs(arr - bg) <= tol, axis=-1)
    arr[mask] = grey
    return Image.fromarray(arr.astype(np.uint8))


def _as_tensor(out) -> torch.Tensor:
    """transformers >=5 returns an output object from get_*_features; older
    versions return a plain tensor. Normalize to a tensor either way."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "last_hidden_state", "image_embeds", "text_embeds"):
        val = getattr(out, attr, None)
        if val is not None:
            return val
    return out[0]


@torch.no_grad()
def _encode_text(prompts: List[str], model, processor, device: str) -> torch.Tensor:
    inputs = processor(text=prompts, return_tensors="pt", padding=True,
                       truncation=True).to(device)
    feat = _as_tensor(model.get_text_features(**inputs))
    return feat / feat.norm(dim=-1, keepdim=True)


@torch.no_grad()
def _encode_images(image_paths: List[Path], model, processor, device: str,
                   mask_bg: bool = False) -> torch.Tensor:
    imgs = []
    for p in image_paths:
        img = Image.open(p)
        if mask_bg:
            img = neutralize_background(img)
        imgs.append(img.convert("RGB"))
    if not imgs:
        return torch.empty(0, device=device)
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    feat = _as_tensor(model.get_image_features(**inputs))
    return feat / feat.norm(dim=-1, keepdim=True)


@torch.no_grad()
def score_images_vs_prompt(image_paths: List[Path], prompt: str, model, preprocess,
                           device: str, mask_bg: bool = False) -> float:
    """Return mean cosine similarity between rendered views and the text prompt."""
    text_feat = _encode_text([prompt], model, preprocess, device)
    img_feats = _encode_images(image_paths, model, preprocess, device, mask_bg=mask_bg)
    if img_feats.numel() == 0:
        return float("nan")
    sims = (img_feats @ text_feat.T).squeeze(-1)
    return float(sims.mean().item())


# Part-noun and position tokens stripped to recover the colour/material descriptor.
# A prompt repeats the same descriptor across many superquadrics (e.g. four legs);
# grouping by descriptor stops high-count parts from outvoting single parts like a
# seat or backrest in the per-attribute mean.
_PART_TOKENS = {
    "leg", "legs", "armrest", "armrests", "rim", "rims", "seat", "backrest",
    "back", "rest", "cushion", "cushioned", "shade", "base", "support", "frame",
    "top", "bottom", "upper", "lower", "left", "right", "middle", "centre",
    "center", "front", "rear", "part", "of", "the", "a", "and",
}


def attribute_key(phrase: str) -> str:
    """Reduce a part phrase to its colour/material descriptor by dropping part-noun
    and position tokens. 'black wrought iron leg upper' -> 'black wrought iron'.
    Falls back to the full phrase if nothing remains."""
    kept = [w for w in phrase.lower().split() if w not in _PART_TOKENS]
    return " ".join(kept) if kept else phrase.lower()


@torch.no_grad()
def score_local_prompts(image_paths: List[Path], local_prompts: Dict[str, str], model,
                        preprocess, device: str, mask_bg: bool = False) -> Dict:
    """Per-attribute CLIP: score each distinct part phrase against the views, then
    average grouped by colour/material descriptor so repeated parts (e.g. four legs)
    count once. This forces CLIP to evaluate each part's colour/material binding
    instead of matching the global prompt as a bag of words.

    Returns {"mean": grouped mean, "per_phrase": {phrase: score},
             "per_attribute": {descriptor: score}}.
    """
    if not local_prompts:
        return {"mean": float("nan"), "per_phrase": {}, "per_attribute": {}}
    img_feats = _encode_images(image_paths, model, preprocess, device, mask_bg=mask_bg)
    if img_feats.numel() == 0:
        return {"mean": float("nan"), "per_phrase": {}, "per_attribute": {}}

    phrases = sorted(set(local_prompts.values()))
    text_feats = _encode_text(phrases, model, preprocess, device)   # [P, D]
    sims = img_feats @ text_feats.T                                 # [views, P]
    per_phrase = {ph: float(sims[:, i].mean().item()) for i, ph in enumerate(phrases)}

    # Group distinct phrases by colour/material descriptor and average within group.
    groups: Dict[str, List[float]] = {}
    for ph, score in per_phrase.items():
        groups.setdefault(attribute_key(ph), []).append(score)
    per_attribute = {k: float(np.mean(v)) for k, v in groups.items()}

    return {"mean": float(np.mean(list(per_attribute.values()))),
            "per_phrase": per_phrase, "per_attribute": per_attribute}


def compute_attribute_winrate(results: Dict[str, list]) -> Dict:
    """Pairwise per-attribute win-rate between exactly two approaches.

    A "win" is one (shape, prompt, attribute-group) where one approach's CLIP score
    is strictly higher; equal scores count as ties. Returns None unless exactly two
    approaches are present.
    """
    approaches = [a for a, recs in results.items() if recs]
    if len(approaches) != 2:
        return {}
    a, b = approaches

    def attr_map(records):
        # (prompt_idx, attribute) -> score
        m = {}
        for r in records:
            for attr, sc in r.get("local_per_attribute", {}).items():
                m[(r["prompt_idx"], attr)] = sc
        return m

    ma, mb = attr_map(results[a]), attr_map(results[b])
    wins = {a: 0, b: 0}
    ties = 0
    for key in ma.keys() & mb.keys():
        if ma[key] > mb[key]:
            wins[a] += 1
        elif mb[key] > ma[key]:
            wins[b] += 1
        else:
            ties += 1
    total = wins[a] + wins[b] + ties
    if total == 0:
        return {}
    return {"approaches": [a, b], "wins": wins, "ties": ties, "total": total}


def find_renders(renders_dir: Path) -> List[Path]:
    pngs = sorted(renders_dir.glob("view_*.png"))
    if not pngs:
        pngs = sorted(renders_dir.glob("*.png"))
    if not pngs:
        pngs = sorted(renders_dir.rglob("*.png"))
    return pngs


# ---------------------------------------------------------------------------

def run_single(args, model, preprocess, device):
    renders_dir = Path(args.renders)
    image_paths = find_renders(renders_dir)
    if not image_paths:
        sys.exit(f"No PNG files found in {renders_dir}")
    score = score_images_vs_prompt(image_paths, args.prompt, model, preprocess, device,
                                   mask_bg=args.mask_bg)
    print(f"CLIP score: {score:.4f}  ({len(image_paths)} views)")
    print(f"Prompt: {args.prompt}")


def run_benchmark(args, model, preprocess, device):
    with open(args.benchmark) as f:
        shapes = json.load(f)

    if getattr(args, "shape_id", None):
        shapes = [s for s in shapes if s["id"] == args.shape_id]
        if not shapes:
            sys.exit(f"No shape with id {args.shape_id} in {args.benchmark}")

    # approach -> list of {shape_id, prompt_idx, prompt, score, n_views}
    results = {}

    for approach in args.approaches:
        print(f"\n{'='*70}")
        print(f"  {approach.upper()}")
        print(f"{'='*70}")
        records = []

        for shape in shapes:
            shape_id = shape["id"]
            category = shape_id.split("_")[0]
            global_desc = shape.get("global_description", "")
            print(f"\n  [{category}] {shape_id}")
            if global_desc:
                print(f"  {global_desc}")
            print(f"  {'-'*66}")

            local_prompts_all = shape.get("local_prompts", [])

            for prompt_idx, prompt in enumerate(shape["prompts"]):
                renders_dir = (Path(args.results_root) / f"{approach}_results"
                               / "renders" / shape_id / f"prompt_{prompt_idx}")
                if not renders_dir.exists():
                    print(f"  prompt {prompt_idx}: [SKIP — no renders dir]")
                    continue
                image_paths = find_renders(renders_dir)
                if not image_paths:
                    print(f"  prompt {prompt_idx}: [SKIP — no PNGs]")
                    continue

                score = score_images_vs_prompt(image_paths, prompt, model, preprocess, device,
                                               mask_bg=args.mask_bg)
                rec = {"shape_id": shape_id, "prompt_idx": prompt_idx,
                       "prompt": prompt, "score": score, "n_views": len(image_paths)}

                local_prompts = (local_prompts_all[prompt_idx]
                                 if prompt_idx < len(local_prompts_all) else {})
                local = score_local_prompts(image_paths, local_prompts, model,
                                            preprocess, device, mask_bg=args.mask_bg)
                rec["local_score"] = local["mean"]
                rec["local_per_phrase"] = local["per_phrase"]
                rec["local_per_attribute"] = local["per_attribute"]

                records.append(rec)
                local_str = (f"  local {local['mean']:.4f}"
                             if not np.isnan(local["mean"]) else "")
                print(f"  [{prompt_idx}] global {score:.4f}{local_str} ({len(image_paths)} views)")
                print(f"       \"{prompt}\"")

        results[approach] = records

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY  (model={args.clip_model}, mask_bg={args.mask_bg})")
    print(f"{'='*70}")
    print(f"  {'Approach':<20} {'Mean CLIP':>10} {'Mean local':>11} {'N':>5}")
    print(f"  {'-'*48}")
    for approach, records in results.items():
        scores = [r["score"] for r in records]
        local_scores = [r["local_score"] for r in records
                        if not np.isnan(r.get("local_score", float("nan")))]
        if scores:
            local_str = f"{np.mean(local_scores):>11.4f}" if local_scores else f"{'N/A':>11}"
            print(f"  {approach:<20} {np.mean(scores):>10.4f} {local_str} {len(scores):>5}")
        else:
            print(f"  {approach:<20} {'N/A':>10} {'N/A':>11} {'0':>5}")

    # Headline metric: per-attribute win-rate, pairwise between two approaches.
    # For each (shape, prompt, attribute-group), the approach with the higher CLIP
    # score wins that group. Win-rate isolates per-part colour/material binding and
    # is robust to CLIP's systematic offset on hard words (e.g. metallic finishes),
    # so it tracks human eval better than the near-tied grouped mean.
    winrate = compute_attribute_winrate(results)
    if winrate:
        a, b = winrate["approaches"]
        tot = winrate["total"]
        print(f"\n  ATTRIBUTE WIN-RATE  ({a} vs {b}, {tot} attribute groups)")
        print(f"  {'-'*48}")
        print(f"  {a:<20} {winrate['wins'][a]:>3} / {tot}  ({winrate['wins'][a]/tot:.0%})")
        print(f"  {b:<20} {winrate['wins'][b]:>3} / {tot}  ({winrate['wins'][b]/tot:.0%})")
        if winrate["ties"]:
            print(f"  {'(ties)':<20} {winrate['ties']:>3} / {tot}")

    if args.output:
        # Per-approach summary
        by_approach = {}
        for approach, records in results.items():
            scores = [r["score"] for r in records]
            local_scores = [r["local_score"] for r in records
                            if not np.isnan(r.get("local_score", float("nan")))]
            by_approach[approach] = {
                "mean": float(np.mean(scores)) if scores else None,
                "min":  float(np.min(scores))  if scores else None,
                "max":  float(np.max(scores))  if scores else None,
                "local_mean": float(np.mean(local_scores)) if local_scores else None,
                "n":    len(scores),
                "records": records,
            }

        # Cross-approach comparison keyed by (shape_id, prompt_idx)
        index: dict = {}
        for approach, records in results.items():
            for r in records:
                key = f"{r['shape_id']}__prompt_{r['prompt_idx']}"
                if key not in index:
                    index[key] = {"shape_id": r["shape_id"],
                                  "prompt_idx": r["prompt_idx"],
                                  "prompt": r["prompt"],
                                  "scores": {},
                                  "local_scores": {}}
                index[key]["scores"][approach] = round(r["score"], 6)
                if not np.isnan(r.get("local_score", float("nan"))):
                    index[key]["local_scores"][approach] = round(r["local_score"], 6)

        # Sort by shape_id then prompt_idx
        comparison = sorted(index.values(), key=lambda x: (x["shape_id"], x["prompt_idx"]))

        out = {"config": {"clip_model": args.clip_model, "mask_bg": args.mask_bg},
               "attribute_winrate": winrate,
               "by_approach": by_approach, "comparison": comparison}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results written to {args.output}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLIP scoring for MultiGen3D benchmark")
    sub = parser.add_subparsers(dest="mode")

    single = sub.add_parser("single", help="Score one renders directory vs one prompt")
    single.add_argument("--renders", required=True)
    single.add_argument("--prompt",  required=True)

    bench = sub.add_parser("benchmark", help="Score all approaches across the full benchmark")
    bench.add_argument("--benchmark",    default="benchmark/prompts_augmented.json")
    bench.add_argument("--results-root", default="results")
    bench.add_argument("--approaches",   nargs="+",
                       default=["multigen", "spacecontrol"])
    bench.add_argument("--output", default="results/clip_scores.json")

    # Flat mode (no subcommand)
    parser.add_argument("--renders",      help="Directory of PNGs (single mode)")
    parser.add_argument("--prompt",       help="Text prompt (single mode)")
    parser.add_argument("--benchmark",    help="Path to prompts.json (benchmark mode)")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--approaches",   nargs="+",
                        default=["multigen", "spacecontrol"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--clip-model", default="ViT-B/32",
                        help="CLIP backbone, e.g. ViT-B/32 or ViT-L/14 (default ViT-B/32)")
    parser.add_argument("--mask-bg", action="store_true",
                        help="Composite pure black/white render backgrounds to neutral grey before scoring")
    parser.add_argument("--shape-id", default=None,
                        help="If set, only score this shape id (benchmark mode)")

    args = parser.parse_args()
    model, preprocess, device = load_clip(args.clip_model)

    if args.renders and args.prompt:
        run_single(args, model, preprocess, device)
    elif args.benchmark:
        run_benchmark(args, model, preprocess, device)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
