"""Absolute Error Score implementation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from flextype import flexdispatch
from probly.representation.array_like import ArrayLike


@flexdispatch
def absolute_error_score[T](
    y_pred: T,
    y_true: T,
) -> npt.NDArray[np.floating]:
    """Compute the absolute error nonconformity score."""
    msg = "Absolute error score computation not implemented for this type."
    raise NotImplementedError(msg)


@absolute_error_score.register(np.ndarray | ArrayLike)
def compute_absolute_error_score_numpy(y_pred: np.ndarray | ArrayLike, y_true: np.ndarray | ArrayLike) -> np.ndarray:
    """Absolute error for numpy arrays."""
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)

    if y_pred_np.ndim == y_true_np.ndim + 1:
        y_pred_np = y_pred_np.mean(axis=0)
    elif y_pred_np.ndim != y_true_np.ndim:
        msg = (
            "y_pred must match y_true shape or add a leading evaluation axis; "
            f"got y_pred shape {y_pred_np.shape} and y_true shape {y_true_np.shape}."
        )
        raise ValueError(msg)

    return np.abs(y_true_np - y_pred_np)
