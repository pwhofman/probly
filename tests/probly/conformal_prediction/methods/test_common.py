"""Tests for common utilities and methods in conformal prediction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from probly.conformal_prediction.methods.common import (
    ConformalPredictor,
    Predictor,
)
from probly.conformal_prediction.utils.quantile import calculate_quantile


class DummyModel(Predictor):
    """Simple dummy model for testing."""

    def __call__(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Return the input as numpy array."""
        return np.array(x, dtype=float)


class DummyConformalPredictor(ConformalPredictor):
    """Test implementation of ConformalPredictor."""

    def predict(self, x_test: Sequence[Any], alpha: float) -> npt.NDArray[np.bool_]:  # noqa: ARG002
        """Test implementation."""
        if not self.is_calibrated or self.threshold is None:
            msg = "Predictor is not calibrated"
            raise RuntimeError(msg)
        # Simple test implementation
        return np.ones((len(x_test), 3), dtype=bool)

    def calibrate(self, x_cal: Sequence[Any], y_cal: Sequence[Any], alpha: float) -> float:
        """Test calibration."""
        if self.conformity_func:
            # Type annotation for the function call
            func: Callable[..., npt.NDArray[np.floating]] = self.conformity_func
            self.nonconformity_scores = func(x_cal, y_cal)
        else:
            # Default: random scores
            rng = np.random.default_rng()
            self.nonconformity_scores = rng.random(len(x_cal))

        # Use shared quantile utility to stay within valid range
        self.threshold = float(calculate_quantile(self.nonconformity_scores, alpha))
        self.is_calibrated = True
        return self.threshold


def test_initial_state_of_conformal_predictor() -> None:
    """Test that ConformalPredictor starts with correct defaults."""
    model = DummyModel()
    cp = DummyConformalPredictor(model)

    assert cp.model is model
    assert cp.conformity_func is None
    assert cp.nonconformity_scores is None
    assert cp.threshold is None
    assert cp.is_calibrated is False


def test_predict_raises_error_if_not_calibrated() -> None:
    """predict() should raise error if called before calibration."""
    model = DummyModel()
    cp = DummyConformalPredictor(model)

    with pytest.raises(RuntimeError):
        cp.predict([1.0, 2.0, 3.0], alpha=0.1)


def test_custom_conformity_function() -> None:
    """Test custom nonconformity function."""
    model = DummyModel()

    def custom_func(
        x: Sequence[Any],
        _y: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Return constant nonconformity scores."""
        return np.ones(len(x), dtype=float) * 3.0

    cp = DummyConformalPredictor(model, nonconformity_func=custom_func)

    # Calibrate with custom function
    x_cal = [0.0, 1.0, 2.0]
    y_cal = [0.0, 0.0, 0.0]
    threshold = cp.calibrate(x_cal, y_cal, alpha=0.2)

    assert cp.is_calibrated
    assert cp.threshold == 3.0
    assert threshold == 3.0
