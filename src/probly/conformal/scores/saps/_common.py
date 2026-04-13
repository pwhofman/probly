"""SAPS Score implementation with optional Randomization (U-term)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.conformal.scores._common import ClassificationNonConformityScore
from probly.representation.distribution import ArrayCategoricalDistribution


@lazydispatch
def saps_score_func[T](
    probs: T,
    y_cal: T | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> npt.NDArray[np.floating]:
    """Compute the SAPS nonconformity scores."""
    msg = "SAPS score computation not implemented for this type."
    raise NotImplementedError(msg)


@saps_score_func.register(np.ndarray)
def _(
    probs: np.ndarray,
    y_cal: np.ndarray | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> np.ndarray:
    """SAPS Nonconformity-Scores for numpy arrays."""
    probs_np = np.asarray(probs, dtype=float)
    n_samples, n_classes = probs_np.shape

    u = np.random.default_rng().uniform(size=(n_samples, n_classes)) if randomized else np.zeros((n_samples, n_classes))

    max_probs = np.max(probs_np, axis=1, keepdims=True)
    sort_idx = np.argsort(-probs_np, axis=1)
    ranks_zero_based = np.argsort(sort_idx, axis=1)
    ranks = ranks_zero_based + 1

    scores = np.where(ranks == 1, u * max_probs, max_probs + (ranks - 2 + u) * lambda_val)

    if y_cal is not None:
        scores = scores[np.arange(n_samples), y_cal]
    return np.asarray(scores, dtype=float)


@saps_score_func.register(ArrayCategoricalDistribution)
def _(
    probs: ArrayCategoricalDistribution,
    y_cal: np.ndarray | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> np.ndarray:
    """SAPS Nonconformity-Scores for ArrayCategoricalDistributions."""
    return saps_score_func(probs.probabilities, y_cal, randomized=randomized, lambda_val=lambda_val)


class SAPSScore[T](ClassificationNonConformityScore[T]):
    non_conformity_score = saps_score_func
