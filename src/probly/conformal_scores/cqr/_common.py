"""CQR (Conformalized Quantile Regression) Score implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike


@flexdispatch
def cqr_score[T](y_pred: T, y_true: T) -> T:
    """Compute the CQR nonconformity score.

    Args:
        y_pred: Predicted lower and upper quantiles, shape ``(n_samples, 2)``.
        y_true: True target values, shape ``(n_samples,)``.

    Returns:
        One-dimensional array of nonconformity scores, shape ``(n_samples,)``.
    """
    msg = f"CQR score computation not implemented for this type. {type(y_pred)}"
    raise NotImplementedError(msg)


@cqr_score.register(np.ndarray)
def compute_cqr_score_numpy(y_pred: np.ndarray | ArrayLike, y_true: np.ndarray) -> np.ndarray:
    """CQR nonconformity scores for numpy arrays."""
    y_np = np.asarray(y_true, dtype=float)
    pred_np = np.asarray(y_pred, dtype=float)

    if pred_np.ndim < y_np.ndim + 1 or pred_np.shape[-1] != 2:
        msg = (
            "y_pred must have shape (..., 2) or (n_evaluations, ..., 2) matching y_true batch shape; "
            f"got y_pred shape {pred_np.shape} and y_true shape {y_np.shape}."
        )
        raise ValueError(msg)

    if pred_np.ndim == y_np.ndim + 2:
        pred_np = pred_np.mean(axis=0)
    elif pred_np.ndim != y_np.ndim + 1:
        msg = (
            "y_pred must match y_true batch rank with a trailing quantile axis, "
            "or include one additional leading evaluation axis."
        )
        raise ValueError(msg)

    lower = pred_np[..., 0]
    upper = pred_np[..., 1]

    return np.maximum(lower - y_np, y_np - upper)


__all__ = ["cqr_score"]
