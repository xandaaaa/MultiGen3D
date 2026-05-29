"""VQA-based per-part colour scoring for the MultiGen3D benchmark.

For each (shape, prompt, attribute-group) we show all rendered views to a VLM
and ask a single direct question:

    "Does the <noun> of this 3D object look <descriptor>? Answer only 'yes' or 'no'."

This avoids the fragile three-step presence→colour→match chain. Whether a part
is structurally present is irrelevant — what matters is whether it *looks* like
the prompt intended.

Per method we report:
  - match_rate : fraction of parts whose colour/material matches the descriptor

Usage:
    python benchmark/vqa_score.py \
        --benchmark benchmark/prompts_augmented.json \
        --results-root results \
        --approaches spacecontrol multigen \
        --output results/vqa_scores.json \
        --api-key sk-...

    # or set OPENAI_API_KEY in the environment and omit --api-key
"""

import argparse
import base64
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from clip_score import attribute_key, find_renders, neutralize_background

DEFAULT_MODEL = "gpt-5.5"

_PART_NOUNS = {
    # furniture
    "leg", "legs", "armrest", "armrests", "rim", "rims", "seat", "backrest",
    "cushion", "shade", "base", "frame", "drawer", "drawers", "door", "doors",
    "handle", "handles", "shelf", "shelves", "panel", "knob", "rail",
    # airplane / vehicle
    "fuselage", "wing", "wings", "tail", "empennage", "cockpit", "nose",
    "engine", "engines", "fin", "rudder", "canopy", "body",
}


def part_noun(phrases: List[str]) -> str:
    """Most common part-noun in the group's phrases (e.g. 'leg', 'fuselage').
    Falls back to the full first phrase when no known noun is found."""
    nouns = [w for ph in phrases for w in ph.lower().split() if w in _PART_NOUNS]
    if not nouns:
        return phrases[0].lower() if phrases else "part"
    noun = Counter(nouns).most_common(1)[0][0]
    return noun.rstrip("s") if noun.endswith("s") else noun


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_client(api_key: str = None):
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OpenAI API key required: pass --api-key or set OPENAI_API_KEY")
    return OpenAI(api_key=key)


def ask(client, model: str, images: List[Image.Image], question: str) -> str:
    content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_image(im)}", "detail": "low"}}
        for im in images
    ]
    content.append({"type": "text", "text": question})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return (resp.choices[0].message.content or "").strip()


def yes(answer: str) -> bool:
    return answer.strip().lower().startswith(("yes", "y "))


def score_group(client, model: str, images: List[Image.Image],
                noun: str, descriptor: str) -> Dict:
    """Ask the VLM directly whether the part matches the intended colour/material."""
    question = (
        f"These are rendered views of a 3D object. "
        f"Does the {noun} look {descriptor}? "
        f"Answer only 'yes' or 'no'."
    )
    answer = ask(client, model, images, question)
    return {"match": yes(answer), "answer": answer}


def run(args):
    with open(args.benchmark) as f:
        shapes = json.load(f)
    if args.shape_id:
        shapes = [s for s in shapes if s["id"] == args.shape_id]
        if not shapes:
            sys.exit(f"No shape with id {args.shape_id}")

    client = load_client(args.api_key)
    model = args.vlm_model
    print(f"Using model: {model}")

    results = {}
    for approach in args.approaches:
        print(f"\n{'='*70}\n  {approach.upper()}\n{'='*70}")
        records = []
        for shape in shapes:
            shape_id = shape["id"]
            local_all = shape.get("local_prompts", [])
            for pi, prompt in enumerate(shape["prompts"]):
                rdir = (Path(args.results_root) / f"{approach}_results"
                        / "renders" / shape_id / f"prompt_{pi}")
                paths = find_renders(rdir)
                if not paths:
                    continue
                images = [neutralize_background(Image.open(p)) for p in paths]
                local = local_all[pi] if pi < len(local_all) else {}

                groups: Dict[str, List[str]] = {}
                for ph in local.values():
                    groups.setdefault(attribute_key(ph), []).append(ph)

                print(f"\n  [{shape_id} prompt {pi}] {prompt}")
                group_results = {}
                for desc, phrases in groups.items():
                    noun = part_noun(phrases)
                    g = score_group(client, model, images, noun, desc)
                    group_results[desc] = {**g, "noun": noun}
                    tag = "MATCH" if g["match"] else "wrong"
                    print(f"     {desc:30s} ({noun:10s}) -> {tag:6s}  [{g['answer']}]")

                records.append({"shape_id": shape_id, "prompt_idx": pi,
                                "prompt": prompt, "groups": group_results})
        results[approach] = records

    # Summary
    print(f"\n\n{'='*70}\n  VQA SUMMARY\n{'='*70}")
    print(f"  {'Approach':<16} {'match_rate':>10} {'parts':>6}")
    print(f"  {'-'*35}")
    summary = {}
    for approach, recs in results.items():
        total = correct = 0
        for r in recs:
            for desc, g in r["groups"].items():
                total += 1
                if g["match"]:
                    correct += 1
        s = {
            "parts": total,
            "match_rate": correct / total if total else None,
        }
        summary[approach] = s
        mr = f"{s['match_rate']:.0%}" if s["match_rate"] is not None else "n/a"
        print(f"  {approach:<16} {mr:>10} {total:>6}")

    if args.output:
        out = {"model": model, "summary": summary, "records": results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Written to {args.output}")


def main():
    ap = argparse.ArgumentParser(description="VQA per-part colour scoring via OpenAI API")
    ap.add_argument("--benchmark", default="benchmark/prompts_augmented.json")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--approaches", nargs="+", default=["spacecontrol", "multigen"])
    ap.add_argument("--shape-id", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--api-key", default=None,
                    help="OpenAI API key (falls back to OPENAI_API_KEY env var)")
    ap.add_argument("--vlm-model", default=DEFAULT_MODEL,
                    help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
