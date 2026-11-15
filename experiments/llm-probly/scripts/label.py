"""Add correctness labels to a generated data file (B1, day 2).

Run from ``experiments/llm-probly``::

    uv run python -m scripts.label --model qwen0.5b --n 200

For every row this

* re-reads the aliases from TriviaQA so that the canonical answer is first
  (no model call is needed, only the dataset),
* adds ``correct_alias`` / ``correct`` for the greedy answer and
  ``samples_correct`` for the 10 samples, using alias matching,
* rewrites ``data/<model>/<dataset>_<n>.jsonl`` and prints the accuracy.
"""

from __future__ import annotations

import argparse

from common.correctness import is_correct, is_correct_letter, matching_alias
from common.data import DATASETS, DEFAULT_SEED, cache_path, load_jsonl, load_questions, save_jsonl


# How an answer is judged, per dataset: alias matching for free text, exact letter for multiple choice.
CORRECTNESS = {"triviaqa": is_correct, "mmlu": is_correct_letter}


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="triviaqa")
    parser.add_argument("--n", type=int, default=50, help="number of questions")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Label every row and print a short summary."""
    args = parse_args()
    path = cache_path(args.model, args.n, args.dataset)
    rows = load_jsonl(path)
    fresh = {q["id"]: q for q in load_questions(args.dataset, args.n, seed=args.seed)}

    for row in rows:
        ref = fresh[row["id"]]
        if ref["question"] != row["question"]:
            msg = f"Question {row['id']} does not match {args.dataset}; was the file made with another seed or dataset?"
            raise ValueError(msg)
        row["aliases"] = ref["aliases"]
        judge = CORRECTNESS[args.dataset]
        row["correct_alias"] = judge(row["greedy_answer"], row["aliases"])
        row["correct"] = row["correct_alias"]
        row["samples_correct"] = [judge(s, row["aliases"]) for s in row["samples"]]

    save_jsonl(rows, path)

    n_correct = sum(r["correct"] for r in rows)
    sample_acc = sum(sum(r["samples_correct"]) for r in rows) / sum(len(r["samples"]) for r in rows)
    print(f"{path.name}: greedy accuracy {n_correct}/{len(rows)} = {n_correct / len(rows):.0%}")
    print(f"sample accuracy (all 10 samples): {sample_acc:.0%}\n")
    for r in rows:
        mark = "OK " if r["correct"] else "-- "
        hit = matching_alias(r["greedy_answer"], r["aliases"])
        extra = f"  [matched: {hit}]" if hit else ""
        print(f"{mark}{r['id']:>2}  {r['greedy_answer'][:60]!r}  | {r['aliases'][0]}{extra}")


if __name__ == "__main__":
    main()
