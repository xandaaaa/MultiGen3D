"""
CLIP scoring for MultiGen3D benchmark.

Usage — score a single directory of renders against a prompt:
    python benchmark/clip_score.py \
        --renders approach1_results/renders/chair_dfeb8d914d8b28ab5bb58f1e92d30bf7/prompt_0/ \
        --prompt "A rounded chair with blue painted metal legs, red velvet seat cushion, ..."

Usage — score all approaches against the full benchmark:
    python benchmark/clip_score.py \
        --benchmark benchmark/prompts.json \
        --results-root . \
        --approaches approach1 approach2 approach5 approach6

Expected renders directory layout (created by the experiment scripts):
    <results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/view_<j>.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image


def load_clip():
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        return model, preprocess, "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        sys.exit("openai-clip not installed. Run: pip install git+https://github.com/openai/CLIP.git")


@torch.no_grad()
def score_images_vs_prompt(image_paths: List[Path], prompt: str, model, preprocess, device: str) -> float:
    """Return mean cosine similarity between rendered views and the text prompt."""
    import clip
    text_tok = clip.tokenize([prompt], truncate=True).to(device)
    text_feat = model.encode_text(text_tok)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    sims = []
    for p in image_paths:
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        img_feat = model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims.append((img_feat @ text_feat.T).item())

    return float(np.mean(sims)) if sims else float("nan")


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
    score = score_images_vs_prompt(image_paths, args.prompt, model, preprocess, device)
    print(f"CLIP score: {score:.4f}  ({len(image_paths)} views)")
    print(f"Prompt: {args.prompt}")


def run_benchmark(args, model, preprocess, device):
    with open(args.benchmark) as f:
        shapes = json.load(f)

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

                score = score_images_vs_prompt(image_paths, prompt, model, preprocess, device)
                records.append({"shape_id": shape_id, "prompt_idx": prompt_idx,
                                 "prompt": prompt, "score": score, "n_views": len(image_paths)})
                print(f"  [{prompt_idx}] {score:.4f} ({len(image_paths)} views)")
                print(f"       \"{prompt}\"")

        results[approach] = records

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Approach':<20} {'Mean CLIP':>10} {'Min':>7} {'Max':>7} {'N':>5}")
    print(f"  {'-'*52}")
    for approach, records in results.items():
        scores = [r["score"] for r in records]
        if scores:
            print(f"  {approach:<20} {np.mean(scores):>10.4f} {np.min(scores):>7.4f} {np.max(scores):>7.4f} {len(scores):>5}")
        else:
            print(f"  {approach:<20} {'N/A':>10} {'':>7} {'':>7} {'0':>5}")

    if args.output:
        # Per-approach summary
        by_approach = {}
        for approach, records in results.items():
            scores = [r["score"] for r in records]
            by_approach[approach] = {
                "mean": float(np.mean(scores)) if scores else None,
                "min":  float(np.min(scores))  if scores else None,
                "max":  float(np.max(scores))  if scores else None,
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
                                  "scores": {}}
                index[key]["scores"][approach] = round(r["score"], 6)

        # Sort by shape_id then prompt_idx
        comparison = sorted(index.values(), key=lambda x: (x["shape_id"], x["prompt_idx"]))

        out = {"by_approach": by_approach, "comparison": comparison}
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
    bench.add_argument("--benchmark",    default="benchmark/prompts.json")
    bench.add_argument("--results-root", default=".")
    bench.add_argument("--approaches",   nargs="+",
                       default=["approach1", "approach2", "approach3",
                                "approach4", "approach5", "approach6"])
    bench.add_argument("--output", default=None)

    # Flat mode (no subcommand)
    parser.add_argument("--renders",      help="Directory of PNGs (single mode)")
    parser.add_argument("--prompt",       help="Text prompt (single mode)")
    parser.add_argument("--benchmark",    help="Path to prompts.json (benchmark mode)")
    parser.add_argument("--results-root", default=".")
    parser.add_argument("--approaches",   nargs="+",
                        default=["approach1", "approach2", "approach3",
                                 "approach4", "approach5", "approach6"])
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    model, preprocess, device = load_clip()

    if args.renders and args.prompt:
        run_single(args, model, preprocess, device)
    elif args.benchmark:
        run_benchmark(args, model, preprocess, device)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
