"""Empirical p(True): let the model answer "(A) True / (B) False" 10 times
(temperature 1.0) and count how often it says True.

Wraps scripts/r3_methods.empirical_p_true() (week 1, R3).
No logits needed, but 10 model calls per question (the most expensive method).
Uncertainty = 1 - p_true (score contract rule 1).
Also stores parsed_fraction per question: how many of the 10 replies could be
read as True or False (the empirical version of coverage).
"""

from __future__ import annotations

import time

import torch

from scripts.r3_methods import empirical_p_true

NAME = "p_true_empirical"
FAMILY = "self_eval"
NEEDS_MODEL = True
N_SAMPLES = 10
SEED = 42   # same seed as the week 1 run (scripts/r3_real_data.py), so results can be reproduced


def score(rows, model=None, tok=None) -> dict:
    """Score the greedy answer of every row with empirical p(True)."""
    if model is None or tok is None:
        raise ValueError(f"{NAME} needs the model (NEEDS_MODEL = True)")

    torch.manual_seed(SEED)   # once, before the loop, exactly as in week 1
    uncertainty, failed, parsed_fraction = [], [], []
    start = time.perf_counter()
    for row in rows:
        p, texts, unparsed = empirical_p_true(model, tok, row["question"], row["greedy_answer"],
                                              n=N_SAMPLES)
        parsed_fraction.append(1 - unparsed / len(texts))
        no_reply_parsed = unparsed == len(texts)
        uncertainty.append(1.0 if no_reply_parsed else 1.0 - p)   # contract rule 3
        failed.append(no_reply_parsed)
    seconds = (time.perf_counter() - start) / len(rows)

    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "probability": True,   # uncertainty = 1 - probability, so summary.py computes ECE
        "parsed_fraction": parsed_fraction,
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": seconds,
        "model_calls_per_question": N_SAMPLES,
    }
