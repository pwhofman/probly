"""Perplexity baseline (BIG).

uncertainty = exp(-mean token log probability of the greedy answer).
Reads the cache only.

Note: exp() keeps the order of -mean log probability, so this ranks the
answers exactly like likelihood_mean (and like lm-polygraph's Perplexity,
which stores -mean log probability without exp). AUROC, AURC and PRR are
therefore identical to likelihood_mean; it is kept under its usual name so the
table lists the standard baselines.
"""

from __future__ import annotations

import math

from methods._cache_baselines import record

NAME = "perplexity"
FAMILY = "baseline"
NEEDS_MODEL = False


def score(rows, model=None, tok=None) -> dict:
    """Uncertainty = perplexity of the greedy answer."""
    values = []
    for r in rows:
        lps = r["greedy_token_logprobs"]
        values.append(math.exp(-sum(lps) / len(lps)) if lps else None)
    return record(NAME, rows, values)
