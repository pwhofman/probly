"""APS Score implementation with optional Randomization (U-term)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from probly.conformal.scores._common import ClassificationNonConformityScore
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch


@lazydispatch
def aps_score_func[T](probs: T, y_cal: T | None = None, randomized: bool = True) -> T:
    """Compute the APS nonconformity scores."""
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
        U = np.random.uniform(low=0, high=1, size=probs_np.shape)
        cumsum_probs -= srt_probs * U

    scores = np.take_along_axis(cumsum_probs, inv_idx, axis=1)
    if y_cal is not None:
        relevant_indices = np.arange(probs_np.shape[0]), y_cal
        scores = scores[relevant_indices]
    return scores


@aps_score_func.register(ArrayCategoricalDistribution)
def _(probs: ArrayCategoricalDistribution, y_cal: np.ndarray | None, randomized: bool = True) -> np.ndarray:
    """APS Nonconformity-Scores for ArrayCategoricalDistributions."""
    return aps_score_func(probs.probabilities, y_cal, randomized=randomized)


class APSScore[T](ClassificationNonConformityScore[T]):
    non_conformity_score = aps_score_func

__all__ = ["aps_score_func", "APSScore"]