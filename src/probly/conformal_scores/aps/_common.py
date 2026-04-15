"""APS Score implementation with optional Randomization (U-term)."""

from __future__ import annotations

import numpy as np

from lazy_dispatch import lazydispatch
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution
from probly.representation.sample.array import ArraySample


@lazydispatch
def aps_score_func[T](probs: T, y_cal: T | None = None, randomized: bool = True) -> T:
    """Compute the APS nonconformity score."""
    msg = "APS score computation not implemented for this type."
    raise NotImplementedError(msg)


@aps_score_func.register(np.ndarray)
def _(probs: np.ndarray, y_cal: np.ndarray | None = None, randomized: bool = True) -> np.ndarray:
    """APS Nonconformity-Scores for numpy arrays."""
    probs_np = np.asarray(probs)

    # sorting indices for descending probabilities
    srt_idx = np.argsort(-probs_np, axis=1)

    # get sorted probabilities
    srt_probs = np.take_along_axis(probs_np, srt_idx, axis=1)

    # calculate cumulative sums
    cumsum_probs = np.cumsum(srt_probs, axis=1)

    # sort back to original positions without in-place writes
    inv_idx = np.argsort(srt_idx, axis=1)

    if randomized:
        u = np.random.default_rng().uniform(low=0, high=1, size=probs_np.shape)
        cumsum_probs -= srt_probs * u

    scores = np.take_along_axis(cumsum_probs, inv_idx, axis=1)
    if y_cal is not None:
        relevant_indices = np.arange(probs_np.shape[0]), y_cal
        scores = scores[relevant_indices]
    return scores


@aps_score_func.register(ArrayCategoricalDistribution)
def _(probs: ArrayCategoricalDistribution, y_cal: np.ndarray | None, randomized: bool = True) -> np.ndarray:
    """APS Nonconformity-Scores for ArrayCategoricalDistributions."""
    return aps_score_func(probs.probabilities, y_cal, randomized=randomized)


@aps_score_func.register(ArraySample)
def _(probs: ArraySample, y_cal: np.ndarray | None, randomized: bool = True) -> np.ndarray:
    """APS Nonconformity-Scores for ArraySamples."""
    return aps_score_func(probs.array, y_cal, randomized=randomized)

__all__ = ["aps_score_func"]
