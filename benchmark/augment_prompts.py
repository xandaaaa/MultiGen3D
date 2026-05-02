"""
Pre-compute per-SQ local prompts for every shape × prompt variant in the benchmark.

Uses GPT-4o to decompose each global styled prompt into one short local prompt
per superquadric, given the part labels in the benchmark JSON.

Output: benchmark/prompts_augmented.json
  Same structure as prompts.json but each entry gains a "local_prompts" key:
    "local_prompts": [
        {0: "a blue painted metal leg", 1: "a blue painted metal leg", ...},  # prompt_0
        {...},   # prompt_1
        ...
    ]

Usage:
    export OPENAI_API_KEY=sk-...
    python benchmark/augment_prompts.py
    python benchmark/augment_prompts.py --input benchmark/prompts.json --output benchmark/prompts_augmented.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI


def decompose_prompt(client, global_prompt: str, parts_dict: dict, global_description: str) -> dict:
    """
    Ask GPT-4o to extract a short per-part local prompt for each SQ.

    Returns {sq_idx (int): local_prompt (str), ...}
    """
    parts_str = "\n".join(f"SQ{k}: {v}" for k, v in sorted(parts_dict.items(), key=lambda x: int(x[0].replace("SQ", ""))  if isinstance(x[0], str) else x[0]))

    user_msg = (
        f'Object type: {global_description}\n\n'
        f'Full styled description: "{global_prompt}"\n\n'
        f'Part labels:\n{parts_str}\n\n'
        'For each SQ part, write a short local prompt (4–8 words) describing ONLY that '
        'part with its specific color and material from the full description. '
        'If the full description does not mention a specific color/material for a part, '
        'infer a neutral but consistent one.\n\n'
        'Return ONLY a JSON object mapping the SQ number (integer key as string) to its local prompt. '
        'Example: {"0": "a blue painted metal chair leg", "1": "a white plastic armrest"}'
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": user_msg}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = json.loads(response.choices[0].message.content)
            # Normalise keys to int (GPT may return "SQ0" or "0")
            result = {}
            for k, v in raw.items():
                k_clean = k.replace("SQ", "").strip()
                result[int(k_clean)] = str(v)
            return result
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)

    return {}


def augment(input_path: str, output_path: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(input_path) as f:
        shapes = json.load(f)

    # Load existing output if present (resume support)
    if Path(output_path).exists():
        with open(output_path) as f:
            augmented = json.load(f)
        done_ids = {s["id"] for s in augmented}
    else:
        augmented = []
        done_ids = set()

    for shape_idx, shape in enumerate(shapes):
        if shape["id"] in done_ids:
            print(f"[{shape_idx+1}/{len(shapes)}] SKIP {shape['id']} (already done)")
            continue

        print(f"\n[{shape_idx+1}/{len(shapes)}] {shape['id']}  ({len(shape['parts'])} SQs, {len(shape['prompts'])} prompts)")

        local_prompts_all = []
        for prompt_idx, prompt in enumerate(shape["prompts"]):
            print(f"  prompt_{prompt_idx}: {prompt[:70]}...")
            per_sq = decompose_prompt(client, prompt, shape["parts"], shape["global_description"])

            # Ensure every SQ index has a fallback
            n_sq = len(shape["parts"])
            for i in range(n_sq):
                if i not in per_sq:
                    per_sq[i] = f"a {shape['global_description'].lower()} part"
                    print(f"    SQ{i} missing — using fallback")

            local_prompts_all.append(per_sq)
            print(f"    -> {per_sq}")
            time.sleep(0.3)  # stay well within rate limits

        entry = dict(shape)
        entry["local_prompts"] = local_prompts_all
        augmented.append(entry)
        done_ids.add(shape["id"])

        # Save after each shape so partial progress is preserved
        with open(output_path, "w") as f:
            json.dump(augmented, f, indent=2)
        print(f"  Saved {output_path} ({len(augmented)}/{len(shapes)} shapes done)")

    print(f"\nDone. Augmented file written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="benchmark/prompts.json")
    parser.add_argument("--output", default="benchmark/prompts_augmented.json")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY before running.")

    augment(args.input, args.output)


if __name__ == "__main__":
    main()
