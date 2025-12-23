"""common utilities for CP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt


class PredictiveModel(Protocol):
    """Protocol for models used with ConformalPredictor."""

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Predict method signature for conformal models."""


class ConformalPredictor:
    """base class for Conformal Prediction."""

    def __init__(
        self,
        model: PredictiveModel,
        nonconformity_func: Callable[..., npt.NDArray[np.floating]] | None = None,
    ) -> None:
        """Initialze the Conformal Predictor."""
        self.model = model
        self.conformity_func = nonconformity_func
        """saves the ML model and nonconformity function"""
        self.nonconformity_scores: npt.NDArray[np.floating] | None = None
        self.threshold: float | None = None
        self.is_calibrated: bool = False

    def __str__(self) -> str:
        """Str representation of the class."""
        model_name = self.model.__class__.__name__
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return f"{self.__class__.__name__}(model={model_name}, status={status})"


class SplitConformal:
    """Utility to split data into training and calibration sets."""

    def __init__(
        self,
        calibration_ratio: float = 0.3,
        random_state: int | None = None,
    ) -> None:
        """Initialize the SplitConformal helper.

        Args:
            calibration_ratio: Fraction of samples used for calibration.
            random_state: Seed for reproducible random splits.
        """
        self.calibration_ratio = calibration_ratio
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.train_indices: npt.NDArray[np.int_] | None = None
        self.cal_indices: npt.NDArray[np.int_] | None = None

    def split(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        calibration_ratio: float | None = None,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """Split data into training and calibration sets."""
        ratio = calibration_ratio if calibration_ratio is not None else self.calibration_ratio

        if not 0 < ratio < 1:
            msg = f"calibration_ratio must be in (0, 1), got {ratio}"
            raise ValueError(msg)

        if len(x) < 2:
            msg = f"Need at least 2 samples, got {len(x)}"
            raise ValueError(msg)

        if len(x) != len(y):
            msg = f"x and y must have the same length. Got x: {len(x)}, y: {len(y)}"
            raise ValueError(msg)

        x = np.asarray(x)
        y = np.asarray(y)

        n_samples = len(x)
        indices = np.arange(n_samples)
        shuffled = self.rng.permutation(indices)

        split_idx = int(n_samples * (1.0 - ratio))
        self.train_indices = shuffled[:split_idx]
        self.cal_indices = shuffled[split_idx:]

        return (
            x[self.train_indices],
            y[self.train_indices],
            x[self.cal_indices],
            y[self.cal_indices],
        )

    def __str__(self) -> str:
        """String representation with basic split information."""
        if self.train_indices is None or self.cal_indices is None:
            return f"SplitConformal(ratio={self.calibration_ratio}, random_state={self.random_state})"

        n_train = len(self.train_indices)
        n_cal = len(self.cal_indices)
        ratio_actual = n_cal / (n_train + n_cal)
        return (
            f"SplitConformal: {n_train} train, {n_cal} calibration "
            f"(ratio={ratio_actual:.3f}, target={self.calibration_ratio})"
        )
