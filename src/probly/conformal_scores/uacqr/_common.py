"""UACQR (Uncertainty-Aware CQR) Score implementation."""

from __future__ import annotations

import numpy as np

from lazy_dispatch import lazydispatch
from probly.representation.sample.array import ArraySample


@lazydispatch
def uacqr_score_func[T](y_pred: T, y_true: T) -> T:
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


@uacqr_score_func.register(np.ndarray)
def compute_uacqr_score_func_numpy(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """UACQR nonconformity scores for numpy arrays."""
    y_true = y_true.flatten()

    if y_pred.ndim != 3 or y_pred.shape[2] != 2:
        msg = f"intervals must have shape (n_estimations, n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    std = np.std(y_pred, axis=0)
    mean_intervals = np.mean(y_pred, axis=0)

    lower = mean_intervals[:, 0]
    upper = mean_intervals[:, 1]
    std_lo = std[:, 0]
    std_hi = std[:, 1]

    return np.maximum((lower - y_true) / std_lo, (y_true - upper) / std_hi)


@uacqr_score_func.register(ArraySample)
def _(y_pred: ArraySample, y_true: np.ndarray) -> np.ndarray:
    """UACQR nonconformity scores for ArraySamples."""
    return compute_uacqr_score_func_numpy(y_pred.array, y_true)
