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
    ConformalClassifier,
    ConformalPredictor,
    ConformalRegressor,
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


def test_calibrate_output_types_and_threshold_range() -> None:
    """Test calibrate method output types and threshold in valid range."""
    model = DummyModel()

    # Test with random scores
    cp_random = DummyConformalPredictor(model)
    x_cal = [1.0, 2.0, 3.0, 4.0]
    y_cal = [0.0, 1.0, 1.0, 2.0]

    threshold_random = cp_random.calibrate(x_cal, y_cal, alpha=0.1)

    # check threshold type and basic range
    assert isinstance(threshold_random, float)
    assert 0 <= threshold_random <= 1  # should be in valid range for random scores
    assert cp_random.threshold == threshold_random

    # Test with fixed scores to verify quantile calculation
    def fixed_scores(_x: Sequence[Any], _y: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Return fixed nonconformity scores."""
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)

    cp_fixed = DummyConformalPredictor(model, nonconformity_func=fixed_scores)

    # with alpha=0.2, should get quantile at 0.8 position
    threshold_fixed = cp_fixed.calibrate([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], alpha=0.2)

    assert isinstance(threshold_fixed, float)
    assert 0.1 <= threshold_fixed <= 0.5  # should be within fixed scores range


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

    # check calibration state
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


# Classification Tests


class DummyConformalClassifier(ConformalClassifier):
    """Test implementation of ConformalClassifier."""

    def predict(self, x_test: Sequence[Any], alpha: float) -> npt.NDArray[np.bool_]:  # noqa: ARG002
        """Generate prediction sets as boolean matrix."""
        # classifier predict implementation
        if not self.is_calibrated or self.threshold is None:
            msg = "Classifier is not calibrated"
            raise RuntimeError(msg)

        # test implementation: return all True for simplicity
        n_samples = len(x_test) if hasattr(x_test, "__len__") else 1
        n_classes = 3  # assume 3 classes
        return np.ones((n_samples, n_classes), dtype=bool)

    def calibrate(self, x_cal: Sequence[Any], y_cal: Sequence[Any], alpha: float) -> float:
        """Calibrate on calibration set."""
        if self.conformity_func:
            func: Callable[..., npt.NDArray[np.floating]] = self.conformity_func
            self.nonconformity_scores = func(x_cal, y_cal)
        else:
            rng = np.random.default_rng()
            self.nonconformity_scores = rng.random(len(x_cal))

        self.threshold = float(calculate_quantile(self.nonconformity_scores, alpha))
        self.is_calibrated = True
        return self.threshold


class DummyConformalRegressor(ConformalRegressor):
    """Test implementation of ConformalRegressor."""

    def predict(self, x_test: Sequence[Any], alpha: float) -> npt.NDArray[np.floating]:  # noqa: ARG002
        """Generate prediction intervals."""
        if not self.is_calibrated or self.threshold is None:
            msg = "Regressor is not calibrated"
            raise RuntimeError(msg)
        # Test implementation: return [y_pred - threshold, y_pred + threshold]
        n_samples = len(x_test) if hasattr(x_test, "__len__") else 1
        predictions = np.linspace(1.0, n_samples, n_samples)
        intervals = np.zeros((n_samples, 2), dtype=float)
        intervals[:, 0] = predictions - self.threshold  # lower
        intervals[:, 1] = predictions + self.threshold  # upper
        return intervals

    def calibrate(self, x_cal: Sequence[Any], y_cal: Sequence[Any], alpha: float) -> float:
        """Calibrate on calibration set."""
        if self.conformity_func:
            func: Callable[..., npt.NDArray[np.floating]] = self.conformity_func
            self.nonconformity_scores = func(x_cal, y_cal)
        else:
            rng = np.random.default_rng()
            self.nonconformity_scores = rng.random(len(x_cal))

        self.threshold = float(calculate_quantile(self.nonconformity_scores, alpha))
        self.is_calibrated = True
        return self.threshold


def test_classifier_predict_output_shape() -> None:
    """Test that ConformalClassifier.predict returns correct shape."""
    model = DummyModel()
    classifier = DummyConformalClassifier(model)

    # calibrate first
    classifier.calibrate([1, 2, 3], [0, 1, 2], alpha=0.1)

    # Test predict with different batch sizes
    x_test_small = [[1], [2]]
    predictions_small = classifier.predict(x_test_small, alpha=0.1)

    assert isinstance(predictions_small, np.ndarray)
    assert predictions_small.dtype == bool  # boolean matrix
    assert predictions_small.shape == (2, 3)  # (n_samples, n_classes)

    # Test with larger batch
    x_test_large = [[1], [2], [3], [4]]
    predictions_large = classifier.predict(x_test_large, alpha=0.1)

    assert predictions_large.shape == (4, 3)
    assert predictions_large.dtype == bool


def test_classifier_predict_requires_calibration() -> None:
    """Test that ConformalClassifier.predict raises error if not calibrated."""
    model = DummyModel()
    classifier = DummyConformalClassifier(model)

    # should raise because not calibrated
    with pytest.raises(RuntimeError, match="not calibrated"):
        classifier.predict([[1], [2]], alpha=0.1)


def test_classifier_is_subclass_of_predictor() -> None:
    """Test that ConformalClassifier is a ConformalPredictor."""
    model = DummyModel()
    classifier = DummyConformalClassifier(model)

    assert isinstance(classifier, ConformalPredictor)
    assert isinstance(classifier, ConformalClassifier)

    # check interface
    assert hasattr(classifier, "predict")
    assert hasattr(classifier, "calibrate")
    assert hasattr(classifier, "is_calibrated")
    assert callable(classifier.predict)
    assert callable(classifier.calibrate)


def test_regressor_predict_output_shape() -> None:
    """Test that ConformalRegressor.predict returns correct shape."""
    model = DummyModel()
    regressor = DummyConformalRegressor(model)

    # calibrate first
    regressor.calibrate([1, 2, 3], [0.5, 1.5, 2.5], alpha=0.1)

    # Test predict with different batch sizes
    x_test_small = [[1], [2]]
    predictions_small = regressor.predict(x_test_small, alpha=0.1)

    assert isinstance(predictions_small, np.ndarray)
    assert np.issubdtype(predictions_small.dtype, np.floating)
    assert predictions_small.shape == (2, 2)  # (n_samples, 2) for [lower, upper]

    # Test with larger batch
    x_test_large = [[1], [2], [3], [4]]
    predictions_large = regressor.predict(x_test_large, alpha=0.1)

    assert predictions_large.shape == (4, 2)
    assert predictions_large.dtype == float

    # check that lower < upper for all intervals
    assert np.all(predictions_large[:, 0] < predictions_large[:, 1])


def test_regressor_predict_requires_calibration() -> None:
    """Test that ConformalRegressor.predict raises error if not calibrated."""
    model = DummyModel()
    regressor = DummyConformalRegressor(model)

    # should raise because not calibrated
    with pytest.raises(RuntimeError, match="not calibrated"):
        regressor.predict([[1], [2]], alpha=0.1)


def test_regressor_is_subclass_of_predictor() -> None:
    """Test that ConformalRegressor is a ConformalPredictor."""
    model = DummyModel()
    regressor = DummyConformalRegressor(model)

    assert isinstance(regressor, ConformalPredictor)
    assert isinstance(regressor, ConformalRegressor)

    # check interface
    assert hasattr(regressor, "predict")
    assert hasattr(regressor, "calibrate")
    assert hasattr(regressor, "is_calibrated")
    assert callable(regressor.predict)
    assert callable(regressor.calibrate)


def test_classifier_and_regressor_different_predict_signatures() -> None:
    """Test that Classifier and Regressor have different predict behaviors."""
    model = DummyModel()
    classifier = DummyConformalClassifier(model)
    regressor = DummyConformalRegressor(model)

    # calibrate both
    classifier.calibrate([1, 2, 3], [0, 1, 2], alpha=0.1)
    regressor.calibrate([1, 2, 3], [0.5, 1.5, 2.5], alpha=0.1)

    x_test = [[1], [2]]

    # classifier returns boolean matrix
    class_preds = classifier.predict(x_test, alpha=0.1)
    assert class_preds.dtype == bool
    assert class_preds.shape == (2, 3)

    # regressor returns floating intervals
    reg_preds = regressor.predict(x_test, alpha=0.1)
    assert np.issubdtype(reg_preds.dtype, np.floating)
    assert reg_preds.shape == (2, 2)
