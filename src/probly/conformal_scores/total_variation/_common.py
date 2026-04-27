"""Total Variation distance nonconformity score implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from flextype import flexdispatch
from probly.conformal_scores import NonConformityScore
from probly.representation.array_like import ArrayLike


@flexdispatch
def tv_score_func[T](y_pred: T, y_true: T | None = None) -> T:
    """Compute Total Variation distance."""
    msg = "Total Variation score not implemented for this type."
    raise NotImplementedError(msg)


@tv_score_func.register(np.ndarray | ArrayLike)
def compute_tv_score_numpy(y_pred: np.ndarray | ArrayLike, y_true: np.ndarray | ArrayLike) -> np.ndarray:
    """Computes the Total Variation score using NumPy Arrays.

    Args:
        y_pred: Probabilities.
        y_true: Calibration prediction
    """
    y_pred_np = np.asarray(y_pred)
    y_true_np = np.asarray(y_true)

    if y_true_np.ndim == 1 or (y_true_np.shape[0] == 1 and y_true_np.size == y_pred_np.shape[0]):
        y_one_hot = np.zeros_like(y_pred_np)
        y_one_hot[np.arange(len(y_true_np)), y_true_np.flatten().astype(int)] = 1.0
        y_true_np = y_one_hot

    return 0.5 * np.sum(np.abs(y_pred_np - y_true_np), axis=-1)

@dataclass(frozen=True, slots=True)
class TVScore[In, Out](NonConformityScore):
    def __call__(self, y_pred: In, y_true: In | None = None) -> Any:
        if y_true is None:
            msg = "y_true is required for TV distance."
            raise ValueError(msg)
        return tv_score_func(y_pred, y_true)

tv_score = TVScore()

__all__ = ["TVScore", "tv_score"]
