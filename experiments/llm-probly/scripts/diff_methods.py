"""Is method A better than method B on this model? Paired bootstrap on one run (E4).

Run from ``experiments/llm-probly``::

    uv run python -m cli diff --run qwen0.5b/triviaqa_1000 likelihood_mean lexical_similarity
    uv run python -m cli diff --run qwen3b/triviaqa_200 p_true_zeroshot msp --metric prr

Both methods are scored on the same resampled questions in every round, and the
95 % interval is taken over the difference. "Tie" means the data cannot tell
the two apart; it does not mean they are equal.
"""

from __future__ import annotations

import argparse

from common.data import cache_path, load_jsonl
from common.evaluate import PairedDifference, paired_bootstrap
from common.io import RESULTS_ROOT, load_scores


def parse_run(run: str) -> tuple[str, str, int]:
    """Split ``<model>/<dataset>_<n>`` into model, dataset and n.

    Raises:
        ValueError: If the run name does not have that form.
    """
    model, _, rest = run.partition("/")
    dataset, _, n = rest.rpartition("_")
    if not model or not dataset or not n.isdigit():
        msg = f"run must look like <model>/<dataset>_<n>, e.g. qwen0.5b/triviaqa_1000; got {run!r}"
        raise ValueError(msg)
    return model, dataset, int(n)


def sentence(name_a: str, name_b: str, result: PairedDifference, run: str) -> str:
    """One-line answer, as it goes into the report."""
    interval = f"95 % interval [{result.ci_low:+.3f}, {result.ci_high:+.3f}]"
    numbers = f"{result.metric.upper()} {result.value_a:.3f} vs {result.value_b:.3f}"
    if result.verdict == "tie":
        return (
            f"On {run}, {name_a} and {name_b} cannot be told apart: {numbers}, "
            f"difference {result.diff:+.3f}, {interval} contains 0."
        )
    m = result.metric.upper()
    if result.verdict == "A":
        better, worse, low, high = name_a, name_b, result.ci_low, result.ci_high
        numbers = f"{m} {result.value_a:.3f} vs {result.value_b:.3f}"
    else:  # B is better: report B's advantage first, so the interval is positive too
        better, worse, low, high = name_b, name_a, -result.ci_high, -result.ci_low
        numbers = f"{m} {result.value_b:.3f} vs {result.value_a:.3f}"
    return (
        f"On {run}, {better} is better than {worse}: {numbers}, by {abs(result.diff):.3f}, "
        f"95 % interval of the advantage [{low:+.3f}, {high:+.3f}] excludes 0."
    )


def main() -> None:
    """Compare two methods of one run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method_a")
    parser.add_argument("method_b")
    parser.add_argument("--run", required=True, help="<model>/<dataset>_<n>, e.g. qwen0.5b/triviaqa_1000")
    parser.add_argument("--metric", default="auroc", choices=["auroc", "prr", "aurc"])
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    model, dataset, n = parse_run(args.run)
    rows = load_jsonl(cache_path(model, n, dataset))
    ids = [row["id"] for row in rows]
    correct = [row["correct"] for row in rows]
    a = load_scores(RESULTS_ROOT / args.run / f"{args.method_a}.json", expected_ids=ids)
    b = load_scores(RESULTS_ROOT / args.run / f"{args.method_b}.json", expected_ids=ids)
    result = paired_bootstrap(a["uncertainty"], b["uncertainty"], correct, metric=args.metric, n_boot=args.n_boot)
    print(sentence(args.method_a, args.method_b, result, args.run))


if __name__ == "__main__":
    main()
