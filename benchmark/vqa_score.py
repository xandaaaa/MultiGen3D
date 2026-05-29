"""VQA-based per-part appearance scoring for the MultiGen3D benchmark.

Motivation: CLIP cosine similarity cannot distinguish a *miscolored* part from an
*absent* one (e.g. the bench armrests that neither approach generates still get a
plausible "light oak armrest" CLIP score). A vision-language model can be asked
directly whether a part exists and what colour/material it is, which is exactly the
judgment human evaluation makes.

For each (shape, prompt, attribute-group) we ask Qwen2.5-VL, shown all rendered
views of the object:
  1. presence : "Is there a <part> in this object?"  -> yes / no
  2. colour   : "What is the colour and material of the <part>?"  -> short phrase
  3. match    : does the colour answer match the prompt's intended descriptor?
                graded by the same model (handles synonyms: navy~dark blue, etc.)

Per method we report, over the parts the prompt asked for:
  - present   : fraction of parts the model says actually exist in the render
  - correct   : fraction of *present* parts whose colour matches the prompt
  - correct_strict : correct over ALL asked parts (absent part counts as wrong)

Usage:
    python benchmark/vqa_score.py \
        --benchmark benchmark/prompts_augmented.json \
        --results-root results \
        --approaches spacecontrol multigen \
        --shape-id bench_f8aa82e7e4c58ce29d31c5ce17cce95d \
        --output results/vqa_scores.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from clip_score import attribute_key, _PART_TOKENS, find_renders, neutralize_background

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Tokens that name the part itself (used to label the group / phrase a question).
_PART_NOUNS = {"leg", "legs", "armrest", "armrests", "rim", "rims", "seat",
               "backrest", "cushion", "shade", "base", "frame"}


def part_noun(phrases: List[str]) -> str:
    """Most common part-noun across a group's phrases, e.g. 'leg' or 'backrest'."""
    nouns = [w for ph in phrases for w in ph.lower().split() if w in _PART_NOUNS]
    if not nouns:
        return "part"
    noun = Counter(nouns).most_common(1)[0][0]
    return noun.rstrip("s") if noun.endswith("s") else noun


def load_vlm():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor, device


@torch.no_grad()
def ask(model, processor, images: List[Image.Image], question: str,
        max_new_tokens: int = 40) -> str:
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def yes(answer: str) -> bool:
    return answer.strip().lower().startswith(("yes", "y "))


def score_group(model, processor, images, noun: str, descriptor: str) -> Dict:
    present_ans = ask(model, processor, images,
                      f"Look at this 3D object. Is there a {noun} present in it? "
                      f"Answer only 'yes' or 'no'.", max_new_tokens=5)
    present = yes(present_ans)
    if not present:
        return {"present": False, "color_answer": None, "match": False,
                "present_answer": present_ans}

    color_ans = ask(model, processor, images,
                    f"What is the colour and material of the {noun} of this object? "
                    f"Answer with a short phrase only.", max_new_tokens=30)
    match_ans = ask(model, processor, images,
                    f"The {noun} is supposed to be '{descriptor}'. "
                    f"You described it as '{color_ans}'. "
                    f"Do these describe the same colour and material? "
                    f"Answer only 'yes' or 'no'.", max_new_tokens=5)
    return {"present": True, "color_answer": color_ans,
            "match": yes(match_ans), "present_answer": present_ans}


def run(args):
    with open(args.benchmark) as f:
        shapes = json.load(f)
    if args.shape_id:
        shapes = [s for s in shapes if s["id"] == args.shape_id]
        if not shapes:
            sys.exit(f"No shape with id {args.shape_id}")

    model, processor, device = load_vlm()
    print(f"Loaded {MODEL_ID} on {device}")

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

                # group phrases by colour/material descriptor
                groups: Dict[str, List[str]] = {}
                for ph in local.values():
                    groups.setdefault(attribute_key(ph), []).append(ph)

                print(f"\n  [{shape_id} prompt {pi}] {prompt}")
                group_results = {}
                for desc, phrases in groups.items():
                    noun = part_noun(phrases)
                    g = score_group(model, processor, images, noun, desc)
                    group_results[desc] = {**g, "noun": noun}
                    tag = ("ABSENT" if not g["present"]
                           else ("MATCH" if g["match"] else "wrong"))
                    ca = g["color_answer"] or ""
                    print(f"     {desc:22s} ({noun:8s}) -> {tag:6s}  \"{ca}\"")

                records.append({"shape_id": shape_id, "prompt_idx": pi,
                                "prompt": prompt, "groups": group_results})
        results[approach] = records

    # ---- summary ----
    print(f"\n\n{'='*70}\n  VQA SUMMARY\n{'='*70}")
    print(f"  {'Approach':<16} {'present':>8} {'correct':>9} {'strict':>8} {'parts':>6}")
    print(f"  {'-'*50}")
    summary = {}
    for approach, recs in results.items():
        total = present = correct = 0
        for r in recs:
            for desc, g in r["groups"].items():
                total += 1
                if g["present"]:
                    present += 1
                    if g["match"]:
                        correct += 1
        s = {
            "parts": total,
            "present_rate": present / total if total else None,
            "correct_of_present": correct / present if present else None,
            "correct_strict": correct / total if total else None,
        }
        summary[approach] = s
        pr = f"{s['present_rate']:.0%}" if s["present_rate"] is not None else "n/a"
        co = f"{s['correct_of_present']:.0%}" if s["correct_of_present"] is not None else "n/a"
        st = f"{s['correct_strict']:.0%}" if s["correct_strict"] is not None else "n/a"
        print(f"  {approach:<16} {pr:>8} {co:>9} {st:>8} {total:>6}")

    if args.output:
        out = {"model": MODEL_ID, "summary": summary, "records": results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Written to {args.output}")


def main():
    ap = argparse.ArgumentParser(description="VQA per-part scoring (Qwen2.5-VL)")
    ap.add_argument("--benchmark", default="benchmark/prompts_augmented.json")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--approaches", nargs="+", default=["spacecontrol", "multigen"])
    ap.add_argument("--shape-id", default=None)
    ap.add_argument("--output", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
