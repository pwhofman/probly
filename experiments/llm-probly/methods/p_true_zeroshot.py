"""Zero-shot p(True): ask the model whether its own greedy answer is true.

Wraps common/reflexive.p_true_details() in plain_ab mode (week 1, R1).
Uncertainty = 1 - p_true (score contract rule 1).
Also stores coverage = P(A) + P(B) per question, to check that the model
really answers with A/B (week 1: ~1.00 on 0.5B; unchecked on larger models).
"""

from __future__ import annotations

import math
import time

from common.reflexive import p_true_details

NAME = "p_true_zeroshot"
FAMILY = "self_eval"
NEEDS_MODEL = True
MODE = "plain_ab"


def score(rows, model=None, tok=None) -> dict:
    """Score the greedy answer of every row with zero-shot p(True)."""
    if model is None or tok is None:
        raise ValueError(f"{NAME} needs the model (NEEDS_MODEL = True)")

    uncertainty, failed, coverage = [], [], []
    start = time.perf_counter()
    for row in rows:
        details = p_true_details(model, tok, row["question"], row["greedy_answer"], mode=MODE)
        p = details["p_true"]
        coverage.append(details["coverage"])   # P(A) + P(B): is the model really answering A/B?
        ok = math.isfinite(p)
        uncertainty.append(1.0 - p if ok else 1.0)   # contract rule 3: failed -> 1.0
        failed.append(not ok)
    seconds = (time.perf_counter() - start) / len(rows)

    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "probability": True,   # uncertainty = 1 - probability, so summary.py computes ECE
        "mode": MODE,
        "coverage": coverage,
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": seconds,
    }
