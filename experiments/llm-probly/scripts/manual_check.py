"""Check automatic correctness labels by hand and estimate the label error rate (DATA).

Run from ``experiments/llm-probly``. Two people split 30 answers per model:

    uv run python -m scripts.manual_check --model qwen0.5b --n 200 --part 1   # DATA, 15 answers
    uv run python -m scripts.manual_check --model qwen0.5b --n 200 --part 2   # BIG, the other 15
    uv run python -m scripts.manual_check --model qwen0.5b --n 200 --summary  # label error rate

The 30 answers are picked once with a fixed seed: half labelled correct by
alias matching and half labelled wrong, so both kinds of label error are
seen. Part 1 and part 2 take every other answer, so they never overlap.
For each answer you see the question, the full answer and the reference
aliases, and type ``y`` (correct) or ``n`` (wrong). Each part is saved to
``data/<model>/<dataset>_<n>.manual_check_part<p>.json``.

Because the check over-samples answers labelled correct, ``--summary``
weights the two groups by how often each label occurs in the whole file
before reporting the overall label error rate.
"""

from __future__ import annotations

import argparse
import json
import random

from common.data import DATASETS, cache_path, load_data

MAX_ALIASES_SHOWN = 8
TOTAL_CHECKED = 30
N_PARTS = 2


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="triviaqa")
    parser.add_argument("--n", type=int, default=200, help="number of questions in the data file")
    parser.add_argument("--seed", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--part", type=int, choices=range(1, N_PARTS + 1), help="which half to check")
    mode.add_argument("--summary", action="store_true", help="combine the parts into a label error rate")
    return parser.parse_args()


def pick_rows(rows: list[dict], k: int, seed: int) -> list[dict]:
    """Pick ``k`` rows, half labelled correct and half incorrect when possible."""
    rng = random.Random(seed)  # noqa: S311 (not used for security)
    right = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    n_right = min(len(right), k // 2)
    n_wrong = min(len(wrong), k - n_right)
    picked = rng.sample(right, n_right) + rng.sample(wrong, n_wrong)
    return sorted(picked, key=lambda r: r["id"])


def part_rows(rows: list[dict], part: int, seed: int) -> list[dict]:
    """Rows of one part: every other row of the fixed 30-row pick."""
    return pick_rows(rows, TOTAL_CHECKED, seed)[part - 1 :: N_PARTS]


def label_error_rate(items: list[dict], share_auto_correct: float) -> dict:
    """Estimate how often the automatic label is wrong on the whole file.

    Args:
        items: Checked answers with boolean ``auto`` and ``manual`` labels.
        share_auto_correct: Fraction of all rows in the file labelled correct automatically.

    Returns:
        Error rate within each label group and the weighted overall error rate.
    """
    groups = {True: [x for x in items if x["auto"]], False: [x for x in items if not x["auto"]]}
    err = {label: (sum(x["auto"] != x["manual"] for x in xs) / len(xs) if xs else 0.0) for label, xs in groups.items()}
    overall = share_auto_correct * err[True] + (1 - share_auto_correct) * err[False]
    return {
        "n_checked": len(items),
        "error_when_labelled_correct": err[True],  # alias matched, but the answer is wrong
        "error_when_labelled_wrong": err[False],  # no alias matched, but the answer is right
        "label_error_rate": overall,
    }


def ask(prompt: str) -> bool:
    """Ask until the user types y or n."""
    while True:
        reply = input(prompt).strip().lower()
        if reply in {"y", "n"}:
            return reply == "y"
        print("Please type y or n.")


def run_part(rows: list[dict], args: argparse.Namespace) -> None:
    """Show one part's answers, collect manual labels and save them."""
    picked = part_rows(rows, args.part, args.seed)
    results = []
    for i, r in enumerate(picked, start=1):
        print(f"\n[{i}/{len(picked)}] id {r['id']}")
        print(f"Question: {r['question']}")
        print(f"Answer:   {r['greedy_answer']}")
        print(f"Aliases:  {r['aliases'][:MAX_ALIASES_SHOWN]}")
        manual = ask("Is the answer correct? (y/n): ")
        results.append({"id": r["id"], "auto": r["correct"], "manual": manual})

    out_path = cache_path(args.model, args.n, args.dataset, suffix=f".manual_check_part{args.part}.json")
    out_path.write_text(json.dumps({"model": args.model, "part": args.part, "items": results}, indent=2) + "\n")

    agree = sum(x["auto"] == x["manual"] for x in results)
    print(f"\nAgreement with the automatic check: {agree}/{len(results)} = {agree / len(results):.0%}")
    for x in results:
        if x["auto"] != x["manual"]:
            print(f"  id {x['id']}: automatic={x['auto']}, you={x['manual']}")
    print(f"Saved to {out_path}")


def run_summary(rows: list[dict], args: argparse.Namespace) -> None:
    """Combine the saved parts into one label error rate."""
    items = []
    for part in range(1, N_PARTS + 1):
        path = cache_path(args.model, args.n, args.dataset, suffix=f".manual_check_part{part}.json")
        if path.exists():
            items += json.loads(path.read_text())["items"]
        else:
            print(f"(part {part} not done yet: {path.name})")
    if not items:
        return
    share = sum(r["correct"] for r in rows) / len(rows)
    result = label_error_rate(items, share)
    print(f"{args.model}: {result['n_checked']} answers checked by hand")
    print(f"  labelled correct, actually wrong: {result['error_when_labelled_correct']:.0%}")
    print(f"  labelled wrong, actually correct: {result['error_when_labelled_wrong']:.0%}")
    print(f"  label error rate on the whole file (weighted): {result['label_error_rate']:.1%}")


def main() -> None:
    """Run one part of the manual check or the summary."""
    args = parse_args()
    rows = load_data(args.model, args.n, args.dataset)
    if args.summary:
        run_summary(rows, args)
    else:
        run_part(rows, args)


if __name__ == "__main__":
    main()
