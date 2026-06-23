"""Kullback-Leibler divergence nonconformity score implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flextype import flexdispatch
from probly.conformal_scores import NonConformityScore
from probly.representation.array_like import ArrayLike


@flexdispatch
def kl_divergence_score_func[T](y_pred: T, y_true: T | None = None) -> T:
    """Compute Kullback-Leibler divergence."""
    msg = "Kullback-Leibler divergence score not implemented for this type."
    raise NotImplementedError(msg)


@kl_divergence_score_func.register(np.ndarray | ArrayLike)
def compute_kl_divergence_score_numpy(y_pred: np.ndarray | ArrayLike, y_true: np.ndarray | ArrayLike) -> np.ndarray:
    """Computes the Kullback-Leibler divergence score using NumPy Arrays.

    Args:
        y_pred: Predicted probabilities.
        y_true: Ground truth labels (class indices or probability vectors).
    """
    y_pred_np = np.asarray(y_pred)
    y_true_np = np.asarray(y_true)

    if y_true_np.ndim == 1 or (y_true_np.shape[0] == 1 and y_true_np.size == y_pred_np.shape[0]):
        y_one_hot = np.zeros_like(y_pred_np)
        y_one_hot[np.arange(len(y_true_np)), y_true_np.flatten().astype(int)] = 1.0
        y_true_np = y_one_hot

    eps = 1e-12
    y_pred_safe = np.clip(y_pred_np, eps, 1.0)
    y_true_safe = np.clip(y_true_np, eps, 1.0)

    return np.sum(y_true_np * np.log(y_true_safe / y_pred_safe), axis=-1)


@dataclass(frozen=True, slots=True)
class KLDivergenceScore[In, Out](NonConformityScore):
    """Kullback-Leibler divergence non-conformity score."""

    def __call__(self, y_pred: In, y_true: In | None = None) -> Any:  # noqa: ANN401
        if y_true is None:
            msg = "y_true is required for Kullback-Leibler divergence."
            raise ValueError(msg)
        return kl_divergence_score_func(y_pred, y_true)


kl_divergence_score = KLDivergenceScore()

__all__ = ["KLDivergenceScore", "kl_divergence_score"]
