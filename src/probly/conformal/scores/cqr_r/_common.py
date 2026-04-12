"""CQR-r (Conformalized Quantile Regression - r) Score implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lazy_dispatch import lazydispatch
from probly.conformal.scores._common import QuantileNonConformityScore
from probly.representation.sample.array import ArraySample

_EPS = 1e-6  # Small constant to prevent division by zero


@lazydispatch
def cqr_r_score_func[T](y_pred: T, y_true: T) -> T:
    """Compute the CQR-r nonconformity score.

    The score is the CQR score normalized by the predicted interval width::

        s(x, y) = max(q_lo - y, y - q_hi) / max(q_hi - q_lo, ε)

    This makes the score adaptive: wider predicted intervals yield smaller
    normalized scores, rewarding models that express higher uncertainty.

    Args:
        y_true: True target values, shape ``(n_samples,)``.
        y_pred: Predicted lower and upper quantiles, shape ``(n_samples, 2)``.

    Returns:
        One-dimensional array of nonconformity scores, shape ``(n_samples,)``.
    """
    msg = "CQR-r score computation not implemented for this type."
    raise NotImplementedError(msg)


@cqr_r_score_func.register(np.ndarray)
def _(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """CQR-r nonconformity scores for numpy arrays."""
    y_np = np.asarray(y_true, dtype=float).reshape(-1)
    pred_np = np.asarray(y_pred, dtype=float)

    if pred_np.ndim != 2 or pred_np.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {pred_np.shape}"
        raise ValueError(msg)

    lower = pred_np[:, 0]
    upper = pred_np[:, 1]
    width = np.maximum(upper - lower, _EPS)

    return np.maximum(lower - y_np, y_np - upper) / width


@cqr_r_score_func.register(ArraySample)
def _(y_pred: ArraySample, y_true: np.ndarray) -> np.ndarray:
    """CQR-r nonconformity scores for ArraySamples."""
    return cqr_r_score_func(y_pred.array, y_true)


@lazydispatch
def weight_func[T](y_pred: T) -> tuple[T, T]:
    """Compute the weights for CQR-r score normalization."""
    msg = "Weight computation not implemented for this type."
    raise NotImplementedError(msg)


@weight_func.register(np.ndarray)
def _(y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weights for CQR-r score normalization for numpy arrays.

    CQR-r calibrates in width-normalized score space (s = residual/width), so
    the test-time adjustment must multiply back by width: [lower - q*width, upper + q*width].
    """
    pred_np = np.asarray(y_pred, dtype=float)

    if pred_np.ndim != 2 or pred_np.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {pred_np.shape}"
        raise ValueError(msg)

    width = np.maximum(pred_np[:, 1] - pred_np[:, 0], _EPS)
    return width, width


@weight_func.register(ArraySample)
def _(y_pred: ArraySample) -> tuple[np.ndarray, np.ndarray]:
    """Weights for CQR-r score normalization for ArraySamples."""
    return weight_func(y_pred.array)


class CQRrScore[T](QuantileNonConformityScore[T]):
    """CQR-r nonconformity score class."""

    non_conformity_score = cqr_r_score_func

    def weight(self, y_pred: T) -> tuple[T, T]:
        """Compute the weights for CQR-r score normalization."""
        return weight_func(y_pred)
