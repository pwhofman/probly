"""Check a few automatic correctness labels by hand (B1, day 2).

Run from ``experiments/llm-probly``::

    uv run python -m scripts.manual_check --model qwen0.5b --n 50 --k 10

Half of the shown questions are labelled correct by alias matching and half
incorrect, picked at random with a fixed seed. For each one you see the
question, the full answer and the reference aliases, and type ``y`` (correct)
or ``n`` (wrong). The result is saved to
``data/<model>/manual_check_<n>.json`` together with the agreement rate.
"""

from __future__ import annotations

import argparse
import json
import random

from common.data import DATA_DIR, load_data

MAX_ALIASES_SHOWN = 8


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--n", type=int, default=50, help="number of questions in the data file")
    parser.add_argument("--k", type=int, default=10, help="how many answers to check by hand")
    parser.add_argument("--seed", type=int, default=0)
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


def ask(prompt: str) -> bool:
    """Ask until the user types y or n."""
    while True:
        reply = input(prompt).strip().lower()
        if reply in {"y", "n"}:
            return reply == "y"
        print("Please type y or n.")


def main() -> None:
    """Show the picked answers, collect manual labels and report agreement."""
    args = parse_args()
    rows = load_data(args.model, args.n)
    picked = pick_rows(rows, args.k, args.seed)

    results = []
    for i, r in enumerate(picked, start=1):
        print(f"\n[{i}/{len(picked)}] id {r['id']}")
        print(f"Question: {r['question']}")
        print(f"Answer:   {r['greedy_answer']}")
        print(f"Aliases:  {r['aliases'][:MAX_ALIASES_SHOWN]}")
        manual = ask("Is the answer correct? (y/n): ")
        results.append({"id": r["id"], "auto": r["correct"], "manual": manual})

    agree = sum(x["auto"] == x["manual"] for x in results)
    summary = {
        "model": args.model,
        "n_checked": len(results),
        "agreement": agree / len(results),
        "items": results,
    }
    out_path = DATA_DIR / args.model / f"manual_check_{args.n}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nAgreement with the automatic check: {agree}/{len(results)} = {agree / len(results):.0%}")
    for x in results:
        if x["auto"] != x["manual"]:
            print(f"  id {x['id']}: automatic={x['auto']}, you={x['manual']}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
