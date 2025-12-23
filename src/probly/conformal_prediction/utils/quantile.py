"""Quantile calculation utility for conformal prediction."""

from __future__ import annotations

import numpy as np


def calculate_quantile(scores: np.ndarray, alpha: float) -> float:
    """Calculate the quantile for conformal prediction.

    Parameters:
    scores : np.ndarray
            Non-conformity scores
    alpha : float
            Significance level (target coverage is 1-alpha)

    Returns:
    float (The (1-alpha)-quantile of the scores)
    """
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # ensure within [0, 1]
    return float(np.quantile(scores, q_level, method="inverted_cdf"))
