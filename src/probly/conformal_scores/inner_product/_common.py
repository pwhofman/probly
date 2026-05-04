"""Inner Product nonconformity score implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flextype import flexdispatch
from probly.conformal_scores import NonConformityScore
from probly.representation.array_like import ArrayLike


@flexdispatch
def inner_product_score_func[T](y_pred: T, y_true: T | None = None) -> T:
    """Compute Inner Product score."""
    msg = "Inner Product score not implemented for this type."
    raise NotImplementedError(msg)


@inner_product_score_func.register(np.ndarray | ArrayLike)
def compute_inner_product_score_numpy(y_pred: np.ndarray | ArrayLike, y_true: np.ndarray | ArrayLike) -> np.ndarray:
    """Computes the Inner Product score using NumPy Arrays.

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

    return 1.0 - np.sum(y_pred_np * y_true_np, axis=-1)


@dataclass(frozen=True, slots=True)
class InnerProductScore[In, Out](NonConformityScore):
    """Inner Product non-conformity score."""

    def __call__(self, y_pred: In, y_true: In | None = None) -> Any:  # noqa: ANN401
        if y_true is None:
            msg = "y_true is required for Inner Product distance."
            raise ValueError(msg)
        return inner_product_score_func(y_pred, y_true)


inner_product_score = InnerProductScore()

__all__ = ["InnerProductScore", "inner_product_score"]
