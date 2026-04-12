"""CQR (Conformalized Quantile Regression) Score implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lazy_dispatch import lazydispatch
from probly.conformal.scores._common import QuantileNonConformityScore
from probly.representation.sample.array import ArraySample

if TYPE_CHECKING:
    from collections.abc import Callable


@lazydispatch
def cqr_score_func[T](y_pred: T, y_true: T) -> T:
    """Compute the CQR nonconformity score.

    Args:
        y_pred: Predicted lower and upper quantiles, shape ``(n_samples, 2)``.
        y_true: True target values, shape ``(n_samples,)``.

    Returns:
        One-dimensional array of nonconformity scores, shape ``(n_samples,)``.
    """
    msg = f"CQR score computation not implemented for this type. {type(y_pred)}"
    raise NotImplementedError(msg)


@cqr_score_func.register(np.ndarray)
def _(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """CQR nonconformity scores for numpy arrays."""
    y_np = np.asarray(y_true, dtype=float).reshape(-1)
    pred_np = np.asarray(y_pred, dtype=float)

    if pred_np.ndim != 2 or pred_np.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {pred_np.shape}"
        raise ValueError(msg)

    lower = pred_np[:, 0]
    upper = pred_np[:, 1]

    return np.maximum(lower - y_np, y_np - upper)


@cqr_score_func.register(ArraySample)
def _(y_pred: ArraySample, y_true: np.ndarray) -> np.ndarray:
    return cqr_score_func(y_pred.samples, y_true)


class CQRScore[T](QuantileNonConformityScore[T]):
    non_conformity_score: Callable[[T, T], T] = cqr_score_func

    def weight[T](self, _: T) -> tuple[T, T]:
        """CQR score does not use weights, so return 1."""
        return 1, 1


__all__ = ["cqr_score_func", "CQRScore"]
