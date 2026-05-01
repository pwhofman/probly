"""Wasserstein distance nonconformity score implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flextype import flexdispatch
from probly.conformal_scores import NonConformityScore
from probly.representation.array_like import ArrayLike


@flexdispatch
def wasserstein_distance_score_func[T](y_pred: T, y_true: T | None = None) -> T:
    """Compute Wasserstein distance."""
    msg = "Wasserstein distance score not implemented for this type."
    raise NotImplementedError(msg)


@wasserstein_distance_score_func.register(np.ndarray | ArrayLike)
def compute_wasserstein_distance_score_numpy(
    y_pred: np.ndarray | ArrayLike, y_true: np.ndarray | ArrayLike
) -> np.ndarray:
    """Computes the Wasserstein distance score using NumPy Arrays.

    Args:
        y_pred: Predicted probability mass functions.
        y_true: True probability mass functions or integer labels.
    """
    y_pred_np = np.asarray(y_pred)
    y_true_np = np.asarray(y_true)

    if y_true_np.ndim == 1 or (y_true_np.shape[0] == 1 and y_true_np.size == y_pred_np.shape[0]):
        y_one_hot = np.zeros_like(y_pred_np)
        y_one_hot[np.arange(len(y_true_np)), y_true_np.flatten().astype(int)] = 1.0
        y_true_np = y_one_hot

    return np.sum(np.abs(np.cumsum(y_pred_np, axis=-1) - np.cumsum(y_true_np, axis=-1)), axis=-1)


@dataclass(frozen=True, slots=True)
class WassersteinDistanceScore[In, Out](NonConformityScore):
    """Wasserstein distance non-conformity score."""

    def __call__(self, y_pred: In, y_true: In | None = None) -> Any:  # noqa: ANN401
        if y_true is None:
            msg = "y_true is required for Wasserstein distance."
            raise ValueError(msg)
        return wasserstein_distance_score_func(y_pred, y_true)


wasserstein_distance_score = WassersteinDistanceScore()

__all__ = ["WassersteinDistanceScore", "wasserstein_distance_score"]
