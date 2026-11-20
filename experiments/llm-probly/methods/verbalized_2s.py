"""Verbalized confidence, two-stage (2S): after the model's own answer, ask
"How confident are you that your answer is correct? Reply with only a number
between 0 and 100." and read the first number in the reply.

Wraps scripts/r3_methods.verbalized_2s() with scale=100 (week 1, R3).
Greedy decoding (do_sample=False), so no seed is needed. 1 short model call per question.
Uncertainty = 1 - confidence; if no number can be read: uncertainty 1.0, failed (contract rule 3).
Also stores the raw reply per question, to check parsing and how many distinct numbers the model uses.
"""

from __future__ import annotations

import time

from scripts.r3_methods import verbalized_2s

NAME = "verbalized_2s"
FAMILY = "self_eval"
NEEDS_MODEL = True
SCALE = 100   # the week 1 run on real data used 0-100


def score(rows, model=None, tok=None) -> dict:
    """Score the greedy answer of every row with verbalized confidence."""
    if model is None or tok is None:
        raise ValueError(f"{NAME} needs the model (NEEDS_MODEL = True)")

    uncertainty, failed, replies = [], [], []
    start = time.perf_counter()
    for row in rows:
        confidence, no_number, reply = verbalized_2s(model, tok, row["question"], row["greedy_answer"],
                                                     scale=SCALE)
        uncertainty.append(1.0 if no_number else 1.0 - confidence)   # contract rule 3
        failed.append(no_number)
        replies.append(reply)
    seconds = (time.perf_counter() - start) / len(rows)

    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "probability": True,   # uncertainty = 1 - probability, so summary.py computes ECE
        "scale": SCALE,
        "replies": replies,
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": seconds,
        "model_calls_per_question": 1,
    }
