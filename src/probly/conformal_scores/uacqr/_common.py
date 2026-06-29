"""UACQR (Uncertainty-Aware CQR) Score implementation."""

from __future__ import annotations

import numpy as np

from flextype import flexdispatch
from probly.representation.array_like import ArrayLike


@flexdispatch
def uacqr_score[T](y_pred: T, y_true: T) -> T:
    """Compute the UACQR nonconformity score.

    Normalizes the CQR score by the standard deviation of the ensemble's
    predicted quantiles across estimations::

        std_lo = std(intervals[:, :, 0], axis=0)
        std_hi = std(intervals[:, :, 1], axis=0)
        q_lo   = mean(intervals[:, :, 0], axis=0)
        q_hi   = mean(intervals[:, :, 1], axis=0)

        s(x, y) = max((q_lo - y) / std_lo, (y - q_hi) / std_hi)

    Args:
        y_pred: Ensemble interval predictions, shape
            ``(n_estimations, n_samples, 2)``, where ``intervals[..., 0]``
            are lower quantiles and ``intervals[..., 1]`` are upper quantiles.
        y_true: True target values, shape ``(n_samples,)``.

    Returns:
        One-dimensional array of nonconformity scores, shape ``(n_samples,)``.
    """
    msg = f"UACQR score computation not implemented for this type {type(y_pred)}."
    raise NotImplementedError(msg)


@uacqr_score.register(np.ndarray | ArrayLike)
def compute_uacqr_score_func_numpy(y_pred: np.ndarray | ArrayLike, y_true: np.ndarray) -> np.ndarray:
    """UACQR nonconformity scores for numpy arrays."""
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)

    if y_pred_np.ndim != y_true_np.ndim + 2 or y_pred_np.shape[-1] != 2:
        msg = (
            "intervals must have shape (n_estimations, ..., 2) with batch shape matching y_true; "
            f"got y_pred shape {y_pred_np.shape} and y_true shape {y_true_np.shape}."
        )
        raise ValueError(msg)

    std = np.std(y_pred_np, axis=0)
    mean_intervals = np.mean(y_pred_np, axis=0)

    lower = mean_intervals[..., 0]
    upper = mean_intervals[..., 1]
    std_lo = std[..., 0]
    std_hi = std[..., 1]

    return np.maximum((lower - y_true_np) / std_lo, (y_true_np - upper) / std_hi)
