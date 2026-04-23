"""Quantile computation utilities for conformal prediction."""

from __future__ import annotations

import numpy as np

from flextype import flexdispatch


@flexdispatch
def calculate_quantile[In](scores: In, alpha: float) -> float:
    """Calculate the conformal quantile from nonconformity scores."""
    msg = "Quantile score computation not implemented for this type."
    raise NotImplementedError(msg)


@flexdispatch
def calculate_weighted_quantile[In](values: In, quantile: float, sample_weight: In | None = None) -> In:
    """Calculate a weighted quantile of the given values."""
    msg = "Weighted quantile computation not implemented for this type."
    raise NotImplementedError(msg)


@calculate_quantile.register(np.ndarray)
def calculate_quantile_numpy(scores: np.ndarray, alpha: float) -> float:
    """Calculate the quantile for conformal prediction.

    Args:
        scores: Non-conformity scores.
        alpha: Significance level (target coverage is 1-alpha).

    Returns:
        The (1-alpha)-quantile of the scores.
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


@calculate_weighted_quantile.register(np.ndarray)
def calculate_weighted_quantile_numpy(
    values: np.ndarray, quantile: float, sample_weight: np.ndarray | None = None
) -> float:
    """Calculate a weighted quantile of the values using numpy."""
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
