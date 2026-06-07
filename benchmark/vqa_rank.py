"""Comparative VLM ranking benchmark for MultiGen3D.

For each (shape, prompt) pair we show a single rendered view from each of two methods
to a VLM and ask it to rank them across 6 criteria, adapted from the SuperDec paper.

Per method we report:
  - avg_rank       : mean rank across all criteria and shape/prompt pairs (lower = better)
  - win_rate       : fraction of (shape, prompt, criterion) triples where rank == 1
  - overall_win    : win_rate on the Overall Quality criterion only

Usage:
    python benchmark/vqa_rank.py \\
        --benchmark benchmark/prompts_augmented.json \\
        --results-root results \\
        --approaches multigen baseline \\
        --output results/vqa_ranking.json \\
        --api-key sk-...

    # or set OPENAI_API_KEY in the environment and omit --api-key
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from clip_score import find_renders, neutralize_background

DEFAULT_MODEL = "gpt-5-mini"

CRITERIA = [
    "Prompt Fidelity",
    "Structure Clarity",
    "Texture Integration",
    "Detail Quality",
    "Part Assignment",
    "Overall Quality",
]

_SYSTEM_PROMPT = """\
Your task is to compare two textured 3D objects generated from the same text prompt \
applied to a 3D mesh. Each method is shown as four rendered views of the same object (front, right, back, left). \
Use all four views together to assess each method — they show the same object from different \
angles and give you a complete picture of how textures are applied across the full surface.

The text prompt specifies the desired appearance: colors, materials, and part-level \
details (e.g. "white fuselage, red-tipped wings, blue tail section"). Evaluate how \
well each method applied textures to match the prompt.

# Criteria

1. Prompt Fidelity: How accurately does the output match the text prompt's color and \
material specifications? Each part should carry its intended appearance. Look for color \
accuracy, material feel (matte, glossy, metallic), and correct part-to-color assignment. \
A strong result maps the right color to the right part — not a random recoloring.

2. Structure Clarity: Does the texture preserve the recognizable geometry of the object? \
Key parts (legs, wings, fuselage, backrest, etc.) should remain distinguishable. Textures \
should enhance, not obscure. Imagine rotating the object: would the structure remain clear? \
Preservation of 3D form and part boundaries is critical.

3. Texture Integration: How smoothly and coherently are textures applied across parts? \
Evaluate transitions at part boundaries, seam visibility, and alignment with geometry. \
Semantically aware integration maps the right texture to the right region. Bad integration \
looks pasted on or mismatched between adjacent parts.

4. Detail Quality: Are local textures (grain, surface finish, patterns) clean, sharp, and \
artifact-free? Look for noise, blur, or visual inconsistencies. Even with simple flat colors, \
the surface should look intentional and uniformly high quality across the mesh.

5. Part Assignment: Does each part of the object carry the correct color or material as \
specified in the prompt? Check that the right appearance is on the right part — e.g., if \
the prompt says "red-tipped wings and blue tail", are the wings red and the tail blue? \
Swapped, merged, or misattributed colors count as failures here even if the overall image \
looks plausible.

6. Overall Quality: Considering all of the above, which output delivers the better result \
overall? Weigh visual appeal, prompt adherence, and technical execution together.

# Output Format

For each criterion, rank the two outputs from best (1) to worst (2). Ties are allowed (1 1). \
Output exactly one line in this format:

Final answer: rankA / rankB / rankC / rankD / rankE / rankF
(Prompt Fidelity / Structure Clarity / Texture Integration / Detail Quality / Part Assignment / Overall)

Each rankX contains two numbers: the rank of Image A followed by the rank of Image B.

Example (Image A wins most criteria, tie on Detail Quality):
Final answer: 1 2 / 1 2 / 2 1 / 1 1 / 1 2 / 1 2\
"""


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


_VIEW_LABELS = ["front", "right", "back", "left"]


def load_views(paths: List[Path]) -> List[Image.Image]:
    """Load and background-neutralize all available views, sorted by view index."""
    ordered = sorted(paths, key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0)
    return [neutralize_background(Image.open(p)) for p in ordered]


def ask_ranking(client, model: str, views_a: List[Image.Image], views_b: List[Image.Image],
                prompt: str) -> str:
    n = len(views_a)
    labels = _VIEW_LABELS[:n] if n <= len(_VIEW_LABELS) else [str(i) for i in range(n)]
    view_desc = " / ".join(labels)

    user_content: list = [{"type": "text", "text": f"Text prompt: {prompt}"}]

    user_content.append({"type": "text", "text": f"\nMethod A — {n} views ({view_desc}):"})
    for img in views_a:
        user_content.append({"type": "image_url",
                              "image_url": {"url": f"data:image/png;base64,{_encode_image(img)}",
                                            "detail": "high"}})

    user_content.append({"type": "text", "text": f"\nMethod B — {n} views ({view_desc}):"})
    for img in views_b:
        user_content.append({"type": "image_url",
                              "image_url": {"url": f"data:image/png;base64,{_encode_image(img)}",
                                            "detail": "high"}})

    user_content.append({"type": "text",
                         "text": "Evaluate and rank the two methods across all six criteria."})

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def parse_ranking(answer: str) -> Optional[List[Tuple[int, int]]]:
    """Parse 'Final answer: 1 2 / 2 1 / ...' into list of (rank_a, rank_b) per criterion."""
    for line in answer.splitlines():
        if "final answer" in line.lower():
            parts_colon = line.split(":", 1)
            if len(parts_colon) < 2:
                continue
            after = parts_colon[1].strip()
            parts = after.split("/")
            if len(parts) != 6:
                continue
            ranks = []
            for part in parts:
                nums = re.findall(r"\d", part)
                if len(nums) == 2:
                    ranks.append((int(nums[0]), int(nums[1])))
            if len(ranks) == 6:
                return ranks
    return None


def run(args):
    with open(args.benchmark) as f:
        shapes = json.load(f)
    if args.shape_id:
        shapes = [s for s in shapes if s["id"] == args.shape_id]
        if not shapes:
            sys.exit(f"No shape with id {args.shape_id}")

    method_a, method_b = args.approaches
    client = load_client(args.api_key)
    model = args.vlm_model
    print(f"Model: {model}  |  {method_a} (A) vs {method_b} (B)")

    records = []
    skipped = 0

    for shape in shapes:
        shape_id = shape["id"]
        for pi, prompt in enumerate(shape["prompts"]):
            dir_a = (Path(args.results_root) / f"{method_a}_results"
                     / "renders" / shape_id / f"prompt_{pi}")
            dir_b = (Path(args.results_root) / f"{method_b}_results"
                     / "renders" / shape_id / f"prompt_{pi}")

            views_a = load_views(find_renders(dir_a))
            views_b = load_views(find_renders(dir_b))
            if not views_a or not views_b:
                skipped += 1
                continue

            print(f"\n[{shape_id}  prompt {pi}]  {prompt[:70]}")
            answer = ask_ranking(client, model, views_a, views_b, prompt)
            ranks = parse_ranking(answer)

            if ranks is None:
                print(f"  WARNING: could not parse ranking:\n  {answer[:300]}")
                skipped += 1
                continue

            for crit, (ra, rb) in zip(CRITERIA, ranks):
                winner = method_a if ra < rb else (method_b if rb < ra else "tie")
                print(f"  {crit:<22}  A={ra}  B={rb}  → {winner}")

            records.append({
                "shape_id": shape_id,
                "prompt_idx": pi,
                "prompt": prompt,
                "ranks": {crit: {"A": ra, "B": rb}
                          for crit, (ra, rb) in zip(CRITERIA, ranks)},
                "raw": answer,
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}\n  RANKING SUMMARY  ({method_a}=A  {method_b}=B)\n{'='*70}")

    if not records:
        print("  No valid comparisons.")
        return

    def aggregate(side: str) -> Dict:
        all_ranks, wins = [], []
        for r in records:
            for crit in CRITERIA:
                rank = r["ranks"][crit][side]
                all_ranks.append(rank)
                wins.append(1 if rank == 1 else 0)
        overall_wins = [1 if r["ranks"]["Overall Quality"][side] == 1 else 0
                        for r in records]
        return {
            "avg_rank": sum(all_ranks) / len(all_ranks),
            "win_rate": sum(wins) / len(wins),
            "overall_win_rate": sum(overall_wins) / len(overall_wins),
        }

    stats = {method_a: aggregate("A"), method_b: aggregate("B")}

    print(f"  {'Method':<18} {'avg_rank':>9} {'win_rate':>9} {'overall_win':>12}")
    print(f"  {'-'*52}")
    for method, s in stats.items():
        print(f"  {method:<18} {s['avg_rank']:>9.2f} {s['win_rate']:>9.0%} "
              f"{s['overall_win_rate']:>12.0%}")

    # Per-criterion breakdown
    print(f"\n  Per-criterion win rates ({method_a} / {method_b}):")
    print(f"  {'-'*52}")
    for crit in CRITERIA:
        wins_a = sum(1 for r in records if r["ranks"][crit]["A"] < r["ranks"][crit]["B"])
        wins_b = sum(1 for r in records if r["ranks"][crit]["B"] < r["ranks"][crit]["A"])
        ties   = sum(1 for r in records if r["ranks"][crit]["A"] == r["ranks"][crit]["B"])
        n = len(records)
        print(f"  {crit:<22}  {method_a}: {wins_a}/{n}  {method_b}: {wins_b}/{n}  ties: {ties}")

    print(f"\n  Skipped: {skipped}  Valid comparisons: {len(records)}")

    if args.output:
        out = {
            "model": model,
            "methods": {"A": method_a, "B": method_b},
            "criteria": CRITERIA,
            "summary": stats,
            "records": records,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Written to {args.output}")


def main():
    ap = argparse.ArgumentParser(description="Comparative VLM ranking for MultiGen3D")
    ap.add_argument("--benchmark", default="benchmark/prompts_augmented.json")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--approaches", nargs=2, default=["multigen", "baseline"],
                    metavar=("METHOD_A", "METHOD_B"))
    ap.add_argument("--shape-id", default=None,
                    help="Only evaluate this shape id")
    ap.add_argument("--output", default=None)
    ap.add_argument("--api-key", default=None,
                    help="OpenAI API key (falls back to OPENAI_API_KEY env var)")
    ap.add_argument("--vlm-model", default=DEFAULT_MODEL,
                    help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
