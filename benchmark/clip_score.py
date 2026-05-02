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
import os
import sys
from pathlib import Path
from typing import List, Optional

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
    """Collect all PNG files in a directory (non-recursive first level)."""
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

    results = {}  # approach -> list of scores

    for approach in args.approaches:
        approach_scores = []
        for shape in shapes:
            shape_id = shape["id"]
            for prompt_idx, prompt in enumerate(shape["prompts"]):
                renders_dir = Path(args.results_root) / f"{approach}_results" / "renders" / shape_id / f"prompt_{prompt_idx}"
                if not renders_dir.exists():
                    print(f"  [SKIP] {approach} / {shape_id} / prompt_{prompt_idx}  (no renders dir)")
                    continue
                image_paths = find_renders(renders_dir)
                if not image_paths:
                    print(f"  [SKIP] {approach} / {shape_id} / prompt_{prompt_idx}  (no PNGs)")
                    continue
                score = score_images_vs_prompt(image_paths, prompt, model, preprocess, device)
                approach_scores.append(score)
                print(f"  {approach:12s}  {shape_id[:30]:30s}  prompt_{prompt_idx}  {score:.4f}")
        results[approach] = approach_scores

    # Summary table
    print("\n" + "=" * 50)
    print(f"{'Approach':<20} {'Mean CLIP':>10} {'N':>6}")
    print("-" * 50)
    for approach, scores in results.items():
        if scores:
            print(f"{approach:<20} {np.mean(scores):>10.4f} {len(scores):>6}")
        else:
            print(f"{approach:<20} {'N/A':>10} {'0':>6}")

    if args.output:
        out = {approach: {"scores": scores, "mean": float(np.mean(scores)) if scores else None}
               for approach, scores in results.items()}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLIP scoring for MultiGen3D benchmark")
    sub = parser.add_subparsers(dest="mode")

    # Single-shot mode
    single = sub.add_parser("single", help="Score one renders directory vs one prompt")
    single.add_argument("--renders", required=True, help="Directory containing PNG renders")
    single.add_argument("--prompt", required=True, help="Text prompt to score against")

    # Full benchmark mode
    bench = sub.add_parser("benchmark", help="Score all approaches across the full benchmark")
    bench.add_argument("--benchmark", default="benchmark/prompts.json", help="Path to prompts.json")
    bench.add_argument("--results-root", default=".", help="Root directory containing <approach>_results/")
    bench.add_argument("--approaches", nargs="+", default=["approach1", "approach2", "approach3", "approach4", "approach5", "approach6"])
    bench.add_argument("--output", default=None, help="Optional JSON file to write results to")

    # Flat mode (no subcommand) for convenience: --renders + --prompt
    parser.add_argument("--renders", help="Directory containing PNG renders (single mode)")
    parser.add_argument("--prompt", help="Text prompt to score against (single mode)")
    parser.add_argument("--benchmark", help="Path to prompts.json (benchmark mode)")
    parser.add_argument("--results-root", default=".", help="Root directory for benchmark mode")
    parser.add_argument("--approaches", nargs="+", default=["approach1", "approach2", "approach3", "approach4", "approach5", "approach6"])
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
