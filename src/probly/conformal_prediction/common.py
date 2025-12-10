"""Implement common utilities and methods for conformal prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.aps.methods.split_conformal import SplitConformal, SplitInfo


class PredictiveModel(Protocol):
    """Protocol for models used with ConformalPredictor."""

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Predict method signature for conformal models."""


class ConformalPredictor(ABC):
    """base class for Conformal Prediction."""

    def __init__(
        self,
        model: PredictiveModel,
        nonconformity_func: Callable[..., npt.NDArray[np.floating]] | None = None,
    ) -> None:
        """Initialze the Conformal Predictor."""
        self.model = model
        self.conformity_func = nonconformity_func
        """saves the ML-model and nonconformity function"""
        self.nonconformity_scores: npt.NDArray[np.floating] | None = None
        self.threshold: float | None = None
        self.is_calibrated: bool = False
        self.splitter: SplitConformal | None = None

    def set_splitter(self, splitter: SplitConformal) -> None:
        """Set a splitter for automatic train/calibration splitting.

        Args:
            splitter: A SplitConformal instance for data splitting
        """
        self.splitter = splitter

    def fit_with_split(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        significance_level: float,
        calibration_ratio: float = 0.3,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Fit the conformal predictor using automatic data splitting.

        Performs split conformal prediction: splits data into training/calibration sets,
        calibrates the predictor, and returns training data for model fitting.

        Args:
            x: Feature data
            y: Label data
            significance_level: Significance level for calibration (e.g., 0.1 for 90% coverage)
            calibration_ratio: Ratio of data to use for calibration
        Returns:
            x_train, y_train: Training data for external model fitting.
        """
        # Ensure arrays
        x_array = np.asarray(x)
        y_array = np.asarray(y)

        # Create splitter if not set
        if self.splitter is None:
            self.splitter = SplitConformal(
                calibration_ratio=calibration_ratio,
            )

        # Split data
        x_train, y_train, x_cal, y_cal = self.splitter.split(
            x_array,
            y_array,
            calibration_ratio=calibration_ratio,
        )

        # Calibrate with calibration data
        self.calibrate(x_cal, y_cal, significance_level)

        # Return training data for external model fitting
        return x_train, y_train

    @abstractmethod
    def _compute_nonconformity(self, x: Sequence[Any], y: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores for given data."""

    @abstractmethod
    def predict(self, x: Sequence[Any], significance_level: float) -> Sequence[Any]:
        """Generate prediction sets for given data at specified significance level."""

    def calibrate(self, x_cal: Sequence[Any], y_cal: Sequence[Any], significance_level: float) -> float | None:
        """Calibrate the conformal predictor using calibration data.

        Args:
            x_cal: Calibration feature data
            y_cal: Calibration label data
            significance_level: Significance level for calibration(e.g., 0.1 for 90% coverage)

        Returns:
            The computed threshold or None if calibration failed.
        """
        self.nonconformity_scores = self._compute_nonconformity(x_cal, y_cal)

        """Stores nonconformity scores for later use in prediction."""

        alpha = significance_level
        self.threshold = float(np.quantile(self.nonconformity_scores, 1 - alpha))

        """Computes and stores the threshold for the given significance level."""

        self.is_calibrated = True

        return self.threshold

    def get_split_info(self) -> SplitInfo | dict[str, Any]:
        """Get information about the last split if splitter was used.

        Returns:
            Dictionary with split information or empty dict if no splitter
        """
        # If no splitter, return empty dict
        if self.splitter is None:
            return {"status": "no splitter configured"}
        return self.splitter.get_split_info()
