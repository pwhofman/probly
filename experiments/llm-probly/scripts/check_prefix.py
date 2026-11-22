"""Check that a larger cache starts with the smaller one (E3: TriviaQA 1000 vs 200).

``common.data`` shuffles the dataset once with a fixed seed and takes the first
``n`` questions, and ``scripts/generate.py`` seeds every question with
``seed + id``. So the first 200 rows of the 1000-question cache must be the
same questions as the 200-question cache, and with the same precision and
device they should also have the same answers. This script checks both, so
results on the 200 prefix stay comparable with last week's numbers.

Run from ``experiments/llm-probly``::

    uv run python -m cli check-prefix --model qwen0.5b --small 200 --large 1000

The exit code is 1 if the questions differ (the caches are not comparable), 2 if
a cache file is missing and 0 otherwise; differing answers are reported but are not an error, because
another device (MPS vs CUDA) can change a few generations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys

from common.data import cache_path, load_jsonl


@dataclass(frozen=True)
class PrefixReport:
    """Result of comparing the first rows of a large cache with a small cache.

    Attributes:
        n_compared: Number of rows compared (the size of the small cache).
        same_questions: True if ids, questions and aliases match row by row.
        first_mismatch: Position of the first row whose question differs, or None.
        same_greedy: Rows with the same greedy answer.
        same_samples: Rows with the same 10 samples, in the same order.
        same_correct: Rows with the same correctness label (only rows labelled in both files).
        n_labelled: Rows labelled in both files.
    """

    n_compared: int
    same_questions: bool
    first_mismatch: int | None
    same_greedy: int
    same_samples: int
    same_correct: int
    n_labelled: int


def compare_prefix(small: list[dict], large: list[dict]) -> PrefixReport:
    """Compare ``small`` with the first rows of ``large``.

    While the large cache is still being generated it can have fewer rows than
    the small one; then only the rows it already has are compared.

    Args:
        small: Rows of the smaller cache.
        large: Rows of the larger cache (or of its ``.partial.jsonl``).

    Returns:
        A report with the counts of matching rows.
    """
    n = min(len(small), len(large))
    pairs = list(zip(small[:n], large[:n], strict=True))

    def same_question(a: dict, b: dict) -> bool:
        return a["id"] == b["id"] and a["question"] == b["question"] and a["aliases"] == b["aliases"]

    mismatches = [i for i, (a, b) in enumerate(pairs) if not same_question(a, b)]
    labelled = [(a, b) for a, b in pairs if "correct" in a and "correct" in b]
    return PrefixReport(
        n_compared=len(pairs),
        same_questions=not mismatches,
        first_mismatch=mismatches[0] if mismatches else None,
        same_greedy=sum(a.get("greedy_answer") == b.get("greedy_answer") for a, b in pairs),
        same_samples=sum(a.get("samples") == b.get("samples") for a, b in pairs),
        same_correct=sum(a["correct"] == b["correct"] for a, b in labelled),
        n_labelled=len(labelled),
    )


def format_report(report: PrefixReport, small_name: str, large_name: str) -> str:
    """Human-readable summary of a ``PrefixReport``."""
    n = report.n_compared
    lines = [f"{large_name}: first {n} rows vs {small_name}"]
    if report.same_questions:
        lines.append(f"  questions: identical ({n}/{n}, same ids, text and aliases)")
    else:
        lines.append(f"  questions: DIFFERENT, first mismatch at row {report.first_mismatch}")
    lines.append(f"  greedy answers identical: {report.same_greedy}/{n}")
    lines.append(f"  samples identical:        {report.same_samples}/{n}")
    if report.n_labelled:
        lines.append(f"  same correctness label:   {report.same_correct}/{report.n_labelled}")
    return "\n".join(lines)


def main() -> None:
    """Compare two cache files of one model and dataset."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="short model name, e.g. qwen0.5b")
    parser.add_argument("--dataset", default="triviaqa")
    parser.add_argument("--small", type=int, default=200, help="size of the existing cache")
    parser.add_argument("--large", type=int, default=1000, help="size of the new cache")
    args = parser.parse_args()

    small_path = cache_path(args.model, args.small, args.dataset)
    large_path = cache_path(args.model, args.large, args.dataset)
    if not large_path.exists():  # a running generation writes .partial.jsonl first
        large_path = cache_path(args.model, args.large, args.dataset, suffix=".partial.jsonl")
    for path in (small_path, large_path):
        if not path.exists():
            print(f"no cache file {path}")
            sys.exit(2)
    report = compare_prefix(load_jsonl(small_path), load_jsonl(large_path))
    print(format_report(report, small_path.name, large_path.name))
    sys.exit(0 if report.same_questions else 1)


if __name__ == "__main__":
    main()
