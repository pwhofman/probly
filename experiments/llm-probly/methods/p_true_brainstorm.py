"""Brainstorm p(True): zero-shot p(True) with the model's own samples in the prompt.

The prompt adds "Here are some ideas that were brainstormed: ..." with the cached samples.

Wraps common/reflexive.p_true_details() in plain_ab mode (week 1, R2).
The samples come from the cache (row["samples"]), so no new generation.
Uncertainty = 1 - p_true (score contract rule 1).
Also stores coverage = P(A) + P(B) per question.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from common.reflexive import p_true_details

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "p_true_brainstorm"
FAMILY = "self_eval"
NEEDS_MODEL = True
MODE = "plain_ab"


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Score the greedy answer of every row with brainstorm p(True)."""
    if model is None or tok is None:
        msg = f"{NAME} needs the model (NEEDS_MODEL = True)"
        raise ValueError(msg)

    uncertainty, failed, coverage = [], [], []
    start = time.perf_counter()
    for row in rows:
        samples = row.get("samples") or []
        if not samples:
            # Without samples the prompt silently becomes zero-shot, so mark it as failed.
            uncertainty.append(1.0)
            failed.append(True)
            coverage.append(float("nan"))
            continue
        details = p_true_details(model, tok, row["question"], row["greedy_answer"], brainstorm=samples, mode=MODE)
        p = details["p_true"]
        coverage.append(details["coverage"])
        ok = math.isfinite(p)
        uncertainty.append(1.0 - p if ok else 1.0)  # contract rule 3: failed -> 1.0
        failed.append(not ok)
    seconds = (time.perf_counter() - start) / len(rows)

    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "probability": True,  # uncertainty = 1 - probability, so summary.py computes ECE
        "mode": MODE,
        "coverage": coverage,
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": seconds,
    }
