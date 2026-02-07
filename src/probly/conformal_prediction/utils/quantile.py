"""Quantile calculation utility for conformal prediction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def calculate_quantile(scores: npt.NDArray[np.floating], alpha: float) -> float:
    """Calculate the quantile for conformal prediction.

    Parameters:
    scores : np.ndarray
            Non-conformity scores
    alpha : float
            Significance level (target coverage is 1-alpha)

    Returns:
    float (The (1-alpha)-quantile of the scores)
    """
    if not 0 <= alpha <= 1:
        msg = f"alpha must be in [0, 1], got {alpha}"
        raise ValueError(msg)

    n = len(scores)
    if n == 0:
        msg = "scores array is empty"
        raise ValueError(msg)

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # ensure within [0, 1]

    # Inverted CDF / right-continuous step quantile
    return float(np.quantile(scores, q_level, method="inverted_cdf"))


def calculate_weighted_quantile(
    values: npt.NDArray[np.floating],
    quantile: float,
    sample_weight: npt.NDArray[np.floating] | None = None,
) -> float:
    """Calculates a weighted quantile of the values using numpy."""
    if sample_weight is None:
        return float(np.quantile(values, quantile, method="higher"))

    values = np.array(values)
    sample_weight = np.array(sample_weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)

    return float(np.interp(quantile, weighted_quantiles, values))
