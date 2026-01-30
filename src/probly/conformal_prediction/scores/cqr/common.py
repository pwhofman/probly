"""Common functions for Conformalized Quantile Regression (CQR) scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.conformal_prediction.methods.common import Predictor

from probly.conformal_prediction.scores.common import RegressionScore


@lazydispatch
def cqr_score_func[T](y_true: T, y_pred: T) -> npt.NDArray[np.floating]:
    """CQR nonconformity scores for generic array types.

    This computes the standard CQR nonconformity score

        s(x_i, y_i) = max(q_lo(x_i) - y_i, y_i - q_hi(x_i), 0),

    where ``y_pred`` contains the predicted lower and upper quantiles
    ``[q_lo, q_hi]`` for each instance.

    Parameters
    ----------
    y_true:
        True target values, shape ``(n_samples,)``.
    y_pred:
        Predicted lower and upper quantiles, shape ``(n_samples, 2)``.

    Returns:
    -------
    np.ndarray
        One-dimensional array of nonconformity scores with shape
        ``(n_samples,)``.
    """
    y_np = np.asarray(y_true, dtype=float).reshape(-1)
    pred_np = np.asarray(y_pred, dtype=float)

    if pred_np.ndim != 2 or pred_np.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {pred_np.shape}"
        raise ValueError(msg)

    lower = pred_np[:, 0]
    upper = pred_np[:, 1]

    # Standard CQR nonconformity: distance to the interval [lower, upper]
    diff_lower = lower - y_np
    diff_upper = y_np - upper

    return np.column_stack([diff_lower, diff_upper])


def register(cls: LazyType, func: Callable[..., Any]) -> None:
    """Register a backend-specific implementation for CQR scores.

    Parameters
    ----------
    cls:
        Lazy type identifying the array backend (e.g. JAX Array).
    func:
        Backend-specific implementation with the same signature as
        :func:`cqr_score_func`.
    """
    cqr_score_func.register(cls=cls, func=func)


class CQRScore(RegressionScore):
    """Backend-agnostic Conformalized Quantile Regression (CQR) score.

    This class wraps :func:`cqr_score_func` and a regression-style
    quantile predictor. The predictor is expected to output, for each
    input instance, a pair ``[q_lo, q_hi]`` representing lower and
    upper conditional quantiles.
    """

    def __init__(self, model: Predictor) -> None:
        """Initialize CQR score with a quantile regression model."""

        def compute_score(
            y_true: npt.NDArray[np.floating], y_pred: npt.NDArray[np.floating]
        ) -> npt.NDArray[np.floating]:
            # Use the same logic as cqr_score_func, but always return shape (N, 1) for predict_nonconformity
            scores: npt.NDArray[np.floating] = cqr_score_func(y_true, y_pred)
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            return scores

        super().__init__(model=model, score_func=compute_score)
