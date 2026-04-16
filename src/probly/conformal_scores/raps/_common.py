"""RAPS Score implementation with optional Randomization (U-term)."""

from __future__ import annotations

import numpy as np

from lazy_dispatch import lazydispatch
from probly.representation.distribution import ArrayCategoricalDistribution
from probly.representation.sample.array import ArraySample


@lazydispatch
def raps_score_func[T](
    probs: T,
    y_cal: T | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> T:
    """Compute the RAPS nonconformity score."""
    msg = "RAPS score computation not implemented for this type."
    raise NotImplementedError(msg)


@raps_score_func.register(np.ndarray)
def compute_raps_score_func_numpy(
    probs: np.ndarray,
    y_cal: np.ndarray | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> np.ndarray:
    """RAPS Nonconformity-Scores for numpy arrays."""
    probs_np = np.asarray(probs, dtype=float)
    n_samples, n_classes = probs_np.shape

    # sorting indices for descending probabilities
    srt_idx = np.argsort(-probs_np, axis=1)
    srt_probs = np.take_along_axis(probs_np, srt_idx, axis=1)

    # calculate cumulative sums
    cumsum_probs = np.cumsum(srt_probs, axis=1)

    if randomized:
        u = np.random.default_rng().uniform(low=0, high=1, size=probs_np.shape)
        cumsum_probs -= srt_probs * u

    # apply regularization: penalize large sets
    ranks = np.arange(1, n_classes + 1).reshape(1, -1)
    penalty = lambda_reg * np.maximum(0, ranks - k_reg - 1)
    epsilon_penalty = epsilon * np.ones((n_samples, n_classes))

    reg_cumsum = cumsum_probs + penalty + epsilon_penalty

    # sort back to original order
    inv_idx = np.argsort(srt_idx, axis=1)
    scores = np.take_along_axis(reg_cumsum, inv_idx, axis=1)

    if y_cal is not None:
        scores = scores[np.arange(n_samples), y_cal]
    return scores


@raps_score_func.register(ArrayCategoricalDistribution)
def _(
    probs: ArrayCategoricalDistribution,
    y_cal: np.ndarray | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> np.ndarray:
    """RAPS Nonconformity-Scores for ArrayCategoricalDistributions."""
    return compute_raps_score_func_numpy(
        probs.probabilities,
        y_cal,
        randomized=randomized,
        lambda_reg=lambda_reg,
        k_reg=k_reg,
        epsilon=epsilon,
    )


@raps_score_func.register(ArraySample)
def _(
    probs: ArraySample,
    y_cal: np.ndarray | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> np.ndarray:
    """RAPS Nonconformity-Scores for ArraySamples."""
    return compute_raps_score_func_numpy(
        probs.array,
        y_cal,
        randomized=randomized,
        lambda_reg=lambda_reg,
        k_reg=k_reg,
        epsilon=epsilon,
    )
