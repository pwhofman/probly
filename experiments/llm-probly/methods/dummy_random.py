"""Dummy method: random uncertainty scores.

Only used to test the pipeline. Its AUROC should be around 0.5 (no signal).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "dummy_random"


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Give every question a random uncertainty between 0 and 1."""
    rng = random.Random(0)  # noqa: S311 - not security related; fixed seed, so the dummy result is the same every run
    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "ids": [row["id"] for row in rows],
        "uncertainty": [rng.random() for _ in rows],
        "failed": [False] * len(rows),
        "seconds_per_question": 0.0,
    }
