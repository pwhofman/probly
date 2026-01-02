"""Tests for common utilities and methods in conformal prediction."""

from __future__ import annotations

import pytest

pytest.importorskip("flax")
pytest.importorskip("jax")

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

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
        # simple test implementation
        return np.ones((len(x_test), 3), dtype=bool)

    def calibrate(self, x_cal: Sequence[Any], y_cal: Sequence[Any], alpha: float) -> float:
        """Test calibration."""
        if self.conformity_func:
            # type annotation for the function call
            func: Callable[..., npt.NDArray[np.floating]] = self.conformity_func
            self.nonconformity_scores = func(x_cal, y_cal)
        else:
            # default: random scores
            rng = np.random.default_rng()
            self.nonconformity_scores = rng.random(len(x_cal))

        # use shared quantile utility to stay within valid range
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

    # calibrate with custom function
    x_cal = [0.0, 1.0, 2.0]
    y_cal = [0.0, 0.0, 0.0]
    threshold = cp.calibrate(x_cal, y_cal, alpha=0.2)

    assert cp.is_calibrated
    assert cp.threshold == 3.0
    assert threshold == 3.0


def test_calibrate_stores_nonconformity_scores() -> None:
    """Test that calibration stores nonconformity scores."""
    model = DummyModel()
    cp = DummyConformalPredictor(model)

    x_cal = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_cal = [0.0, 0.0, 1.0, 1.0, 2.0]

    cp.calibrate(x_cal, y_cal, alpha=0.1)

    assert cp.nonconformity_scores is not None
    assert len(cp.nonconformity_scores) == 5
    assert isinstance(cp.nonconformity_scores, np.ndarray)


def test_predict_after_calibration() -> None:
    """Test that predict works after calibration."""
    model = DummyModel()
    cp = DummyConformalPredictor(model)

    # calibrate first
    x_cal = [1.0, 2.0, 3.0]
    y_cal = [0.0, 0.0, 0.0]
    cp.calibrate(x_cal, y_cal, alpha=0.1)

    # predict now
    x_test = [4.0, 5.0]
    prediction_sets = cp.predict(x_test, alpha=0.1)

    assert prediction_sets is not None
    assert isinstance(prediction_sets, np.ndarray)
    assert prediction_sets.dtype == bool
    assert prediction_sets.shape == (2, 3)


def test_calibrate_output_types() -> None:
    """Test calibrate method output types."""
    model = DummyModel()
    cp = DummyConformalPredictor(model)

    x_cal = [1.0, 2.0, 3.0, 4.0]
    y_cal = [0.0, 1.0, 1.0, 2.0]

    threshold = cp.calibrate(x_cal, y_cal, alpha=0.1)

    # check threshold type
    assert isinstance(threshold, float)
    assert 0 <= threshold <= 1  # should be in valid range for random scores
    assert cp.threshold == threshold


def test_predict_output_types() -> None:
    """Test predict method output types."""
    model = DummyModel()
    cp = DummyConformalPredictor(model)

    # calibrate
    cp.calibrate([1.0, 2.0], [0.0, 1.0], alpha=0.1)

    # predict
    prediction_sets = cp.predict([3.0, 4.0, 5.0], alpha=0.1)

    assert isinstance(prediction_sets, np.ndarray)
    assert prediction_sets.dtype == bool
    assert prediction_sets.shape == (3, 3)


def test_edge_case_single_sample_calibration() -> None:
    """Test calibration with single sample."""
    model = DummyModel()
    cp = DummyConformalPredictor(model)

    x_cal = [1.0]
    y_cal = [0.0]

    threshold = cp.calibrate(x_cal, y_cal, alpha=0.1)

    assert cp.is_calibrated
    assert isinstance(threshold, float)
    assert cp.nonconformity_scores is not None
    assert len(cp.nonconformity_scores) == 1


def test_edge_case_large_batch_calibration() -> None:
    """Test calibration with large batch."""
    model = DummyModel()
    cp = DummyConformalPredictor(model)

    n_samples = 1000
    x_cal = list(range(n_samples))
    y_cal = [i % 3 for i in range(n_samples)]

    threshold = cp.calibrate(x_cal, y_cal, alpha=0.1)

    assert cp.is_calibrated
    assert isinstance(threshold, float)
    assert cp.nonconformity_scores is not None
    assert len(cp.nonconformity_scores) == n_samples


def test_edge_case_extreme_alpha_values() -> None:
    """Test calibration with extreme alpha values."""
    model = DummyModel()

    # Test with very small alpha (strict)
    cp_strict = DummyConformalPredictor(model)
    threshold_strict = cp_strict.calibrate([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], alpha=0.01)
    assert isinstance(threshold_strict, float)

    # Test with large alpha (permissive)
    cp_permissive = DummyConformalPredictor(model)
    threshold_permissive = cp_permissive.calibrate([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], alpha=0.5)
    assert isinstance(threshold_permissive, float)


def test_threshold_in_valid_range() -> None:
    """Test that threshold is computed correctly and in valid range."""
    model = DummyModel()

    def fixed_scores(_x: Sequence[Any], _y: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Return fixed nonconformity scores."""
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

    cp = DummyConformalPredictor(model, nonconformity_func=fixed_scores)

    # with alpha=0.2, should get quantile at 0.8 position
    threshold = cp.calibrate([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], alpha=0.2)

    assert 0.1 <= threshold <= 0.5
    assert isinstance(threshold, float)
