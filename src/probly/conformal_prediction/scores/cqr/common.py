"""Common functions for Conformalized Quantile Regression (CQR) scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lazy_dispatch.isinstance import LazyType
    from probly.conformal_prediction.methods.common import Predictor


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
    zeros = np.zeros_like(diff_lower)

    scores = np.maximum.reduce((diff_lower, diff_upper, zeros))
    return scores.astype(float)


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


class CQRScore:
    """Backend-agnostic Conformalized Quantile Regression (CQR) score.

    This class wraps :func:`cqr_score_func` and a regression-style
    quantile predictor. The predictor is expected to output, for each
    input instance, a pair ``[q_lo, q_hi]`` representing lower and
    upper conditional quantiles.
    """

    def __init__(self, model: Predictor) -> None:
        """Initialize CQR score with a quantile regression model."""
        self.model = model

    def _predict_intervals(
        self,
        x: Sequence[Any],
        y_pred: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Return predicted quantile intervals as a NumPy array."""
        # Fixes SIM108: Use ternary operator
        raw_pred = self.model(x) if y_pred is None else y_pred

        pred_np = np.asarray(raw_pred, dtype=float)
        if pred_np.ndim != 2 or pred_np.shape[1] != 2:
            msg = f"Model outputs for CQR must have shape (n_samples, 2), got {pred_np.shape}"
            raise ValueError(msg)
        return pred_np

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        y_pred: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute 1D CQR nonconformity scores on calibration data.

        Parameters
        ----------
        x_cal:
            Calibration inputs.
        y_cal:
            Calibration targets.
        y_pred:
            Optional pre-computed predicted intervals, shape ``(n_samples, 2)``.
            If ``None``, they are obtained from ``self.model``.
        """
        intervals = self._predict_intervals(x_cal, y_pred=y_pred)
        y_np = np.asarray(y_cal, dtype=float).reshape(-1)

        if len(y_np) != intervals.shape[0]:
            msg = f"y_cal and predicted intervals must have same length, got {len(y_np)} and {intervals.shape[0]}"
            raise ValueError(msg)

        scores: npt.NDArray[np.floating] = cqr_score_func(y_np, intervals)
        return np.asarray(scores, dtype=float)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        y_pred: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Return a 2D score matrix for compatibility with the Score protocol.

        For regression, there is typically a single target dimension, so we
        expose a single-column score matrix whose entries are the widths of
        the predicted intervals, ``q_hi - q_lo``. This keeps the API shape
        consistent with other scores while remaining meaningful for regression.
        """
        intervals = self._predict_intervals(x_test, y_pred=y_pred)
        widths = intervals[:, 1] - intervals[:, 0]
        widths = np.asarray(widths, dtype=float).reshape(-1, 1)
        return widths
