"""Common structures for conformal prediction scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt


class Score(Protocol):
    """Interface for nonconformity scores used in split conformal prediction.

    Each score (APS, LAC, RAPS, ...) must implement:
    - calibration_nonconformity: used on calibration data.
    - predict_nonconformity: used on test data, must return a score matrix
      of shape according to the specific score type (classification or regression).
    """

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Return 1D array of nonconformity scores for calibration instances."""


class ClassificationScore(Score, Protocol):
    """Nonconformity scores for classification tasks.

    calibration_nonconformity: 1D scores from Score.
    predict_nonconformity: 2D scores (n_instances, n_labels).
    """

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: Any = None,  # noqa: ANN401
    ) -> npt.NDArray[np.floating]:
        """Return 2D score matrix of shape (n_instances, n_labels)."""


class RegressionScore(Score, Protocol):
    """Nonconformity scores for regression (e.g.. Residuals).

    calibration_nonconformity: 1D scores (|y - y_hat|, standardized Residuals, ...).
    predict_nonconformity: 1D scores or local scales (n_instances,).
    """

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Return 1D scores or scales of shape (n_instances,)."""

    def construct_intervals(
        self,
        y_hat: npt.NDArray[np.floating],
        threshold: float,
    ) -> npt.NDArray[np.floating]:
        """Construct prediction intervals based on model output and threshold.

        Args:
            y_hat: Model output (n_samples, ) or (n_samples, 2) etc.
            threshold: Calibrated q-hat.

        Returns:
            Intervals as (n_samples, 2) matrix [lower, upper].
        """
        ...
