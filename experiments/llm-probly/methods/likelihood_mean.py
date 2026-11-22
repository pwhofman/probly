"""Likelihood baseline: how unlikely the model found its own greedy answer.

uncertainty = -mean token log likelihood of the greedy answer (length-normalised).
Reads the cache only, so it needs no model and costs nothing extra.
Plugin version of scripts/baseline_likelihood.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "likelihood_mean"


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Uncertainty = minus the mean log likelihood of the greedy answer."""
    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "ids": [row["id"] for row in rows],
        "uncertainty": [-row["greedy_ll_mean"] for row in rows],
        "failed": [False] * len(rows),
        "seconds_per_question": 0.0,  # computed during generation, no extra cost
    }
