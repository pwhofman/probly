"""Likelihood baselines from the stored generation data (B1).

The simplest uncertainty score needs no extra model call: how unlikely the
model found its own greedy answer. Two variants are saved in the score
contract format (see CONTRACT.md):

* ``likelihood_mean``: uncertainty = -mean token log likelihood (length-normalised)
* ``likelihood_sum``: uncertainty = -summed log likelihood (longer answers look more uncertain)

Run from ``experiments/llm-probly``::

    uv run python -m scripts.baseline_likelihood
"""

from __future__ import annotations

from common.data import load_data
from common.io import save_scores

VARIANTS = {
    "likelihood_mean": "greedy_ll_mean",
    "likelihood_sum": "greedy_ll_sum",
}


def main() -> None:
    """Build both baselines and save them under results/qwen0.5b/."""
    rows = load_data("qwen0.5b", 50)
    ids = [r["id"] for r in rows]
    for method, field in VARIANTS.items():
        record = {
            "method": method,
            "model": rows[0]["model"],
            "answer": "greedy",
            "ids": ids,
            "uncertainty": [-r[field] for r in rows],
            "failed": [False] * len(rows),
            "seconds_per_question": 0.0,  # computed during generation, no extra cost
        }
        path = save_scores(record, expected_ids=ids)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
