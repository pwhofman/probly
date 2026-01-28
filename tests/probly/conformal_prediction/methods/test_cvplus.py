"""Tests for CV+ Conformal Prediction with Type Annotations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from probly.conformal_prediction.methods.cvplus import (
    CVPlusClassifier,
    CVPlusRegressor,
)
from probly.conformal_prediction.methods.jackknife import (
    JackknifePlusClassifier,
    JackknifePlusRegressor,
)


class MockSklearnModel:
    """Mock sklearn-like model for CV+ testing."""

    def __init__(self, output_dim: int = 1) -> None:
        """Initialize mock model with output dimension."""
        self.output_dim = output_dim
        self.is_fitted = False
        self.x_shape: tuple[int, ...] | None = None
        self.y_shape: tuple[int, ...] | None = None

    def fit(self, x: npt.NDArray[Any], y: npt.NDArray[Any]) -> None:
        """Fit mock model."""
        self.is_fitted = True
        self.x_shape = x.shape
        self.y_shape = y.shape

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[np.floating]:
        """Predict with mock model."""
        if not self.is_fitted:
            error_msg = "Model not fitted"
            raise RuntimeError(error_msg)

        n = len(x)
        if self.output_dim == 1:
            # regression: return constant
            return np.ones(n, dtype=float) * 0.5
        # classification: return uniform probabilities
        return np.ones((n, self.output_dim), dtype=float) / self.output_dim

    def __call__(self, x: Sequence[Any] | npt.NDArray[Any]) -> npt.NDArray[np.floating]:
        """Call mock model."""
        x_arr = np.asarray(x)
        return self.predict(x_arr)


def regression_model_factory_func() -> MockSklearnModel:
    """Factory for regression models."""
    return MockSklearnModel(output_dim=1)


def classification_model_factory_func() -> MockSklearnModel:
    """Factory for classification models."""
    return MockSklearnModel(output_dim=3)


@pytest.fixture
def regression_model_factory() -> Callable[[], MockSklearnModel]:
    """Factory for regression models."""
    return regression_model_factory_func


@pytest.fixture
def classification_model_factory() -> Callable[[], MockSklearnModel]:
    """Factory for classification models."""
    return classification_model_factory_func


class TestCVPlusRegressor:
    """Tests for CVPlusRegressor."""

    def test_initialization_default_cv(self, regression_model_factory: Callable[[], Any]) -> None:
        """Test initialization with default CV (5-fold)."""
        regressor = CVPlusRegressor(
            model_factory=regression_model_factory,
            random_state=42,
        )

        assert regressor.model_factory is regression_model_factory
        assert regressor.cv == 5  # default
        assert regressor.random_state == 42
        assert not regressor.is_calibrated

    def test_initialization_custom_cv(self, regression_model_factory: Callable[[], Any]) -> None:
        """Test initialization with custom CV."""
        regressor = CVPlusRegressor(
            model_factory=regression_model_factory,
            cv=10,
            random_state=123,
        )

        assert regressor.cv == 10
        assert regressor.random_state == 123

    def test_calibration(self, regression_model_factory: Callable[[], Any]) -> None:
        """Test calibration with CV+."""
        rng = np.random.default_rng(42)
        x = rng.random((50, 5))
        y = rng.random(50)

        regressor = CVPlusRegressor(
            model_factory=regression_model_factory,
            cv=5,
            random_state=42,
        )

        alpha = 0.1
        threshold = regressor.calibrate(x.tolist(), y.tolist(), alpha)

        assert regressor.is_calibrated
        assert threshold is not None
        assert isinstance(threshold, float)

        # should have 5 models (5-fold CV)
        assert len(regressor.fitted_models) == 5
        assert regressor.n_folds_actual_ == 5

    def test_prediction(self, regression_model_factory: Callable[[], Any]) -> None:
        """Test prediction intervals."""
        rng = np.random.default_rng(42)
        x = rng.random((50, 5))
        y = rng.random(50)

        regressor = CVPlusRegressor(
            model_factory=regression_model_factory,
            cv=5,
            random_state=42,
        )

        # calibrate
        regressor.calibrate(x.tolist(), y.tolist(), alpha=0.1)

        # predict on test data
        x_test = rng.random((10, 5))
        intervals = regressor.predict(x_test.tolist(), alpha=0.1)

        assert intervals.shape == (10, 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])

    def test_different_cv_values(self, regression_model_factory: Callable[[], Any]) -> None:
        """Test with different CV values."""
        rng = np.random.default_rng(42)
        x = rng.random((60, 5))
        y = rng.random(60)

        for cv in [2, 3, 5, 10]:
            regressor = CVPlusRegressor(
                model_factory=regression_model_factory,
                cv=cv,
                random_state=42,
            )

            regressor.calibrate(x.tolist(), y.tolist(), alpha=0.1)
            assert regressor.is_calibrated
            assert regressor.n_folds_actual_ == min(cv, len(x))

            # should have created correct number of models
            assert len(regressor.fitted_models) == regressor.n_folds_actual_


class TestCVPlusClassifier:
    """Tests for CVPlusClassifier."""

    def test_initialization_default_cv(self, classification_model_factory: Callable[[], Any]) -> None:
        """Test initialization with default CV (5-fold)."""
        classifier = CVPlusClassifier(
            model_factory=classification_model_factory,
            random_state=42,
            use_accretive=False,
        )

        assert classifier.model_factory is classification_model_factory
        assert classifier.cv == 5  # default
        assert classifier.random_state == 42
        assert classifier.use_accretive is False
        assert not classifier.is_calibrated

    def test_calibration(self, classification_model_factory: Callable[[], Any]) -> None:
        """Test calibration with CV+."""
        rng = np.random.default_rng(42)
        x = rng.random((50, 5))
        y = rng.integers(0, 3, 50)

        classifier = CVPlusClassifier(
            model_factory=classification_model_factory,
            cv=5,
            random_state=42,
        )

        alpha = 0.1
        threshold = classifier.calibrate(x.tolist(), y.tolist(), alpha)

        assert classifier.is_calibrated
        assert threshold is not None
        assert isinstance(threshold, float)

        # should have learned classes
        assert classifier.classes is not None
        assert len(classifier.classes) == 3

    def test_prediction(self, classification_model_factory: Callable[[], Any]) -> None:
        """Test prediction sets."""
        rng = np.random.default_rng(42)
        x = rng.random((50, 5))
        y = rng.integers(0, 3, 50)

        classifier = CVPlusClassifier(
            model_factory=classification_model_factory,
            cv=5,
            random_state=42,
        )

        # calibrate
        classifier.calibrate(x.tolist(), y.tolist(), alpha=0.1)

        # predict on test data
        x_test = rng.random((10, 5))
        prediction_sets = classifier.predict(x_test.tolist(), alpha=0.1)

        assert prediction_sets.shape == (10, 3)
        assert prediction_sets.dtype == np.bool_
        assert np.all(prediction_sets.sum(axis=1) >= 1)

    def test_prediction_with_accretive(self, classification_model_factory: Callable[[], Any]) -> None:
        """Test prediction with accretive completion."""
        rng = np.random.default_rng(42)
        x = rng.random((50, 5))
        y = rng.integers(0, 3, 50)

        classifier = CVPlusClassifier(
            model_factory=classification_model_factory,
            cv=5,
            random_state=42,
            use_accretive=True,
        )

        classifier.calibrate(x.tolist(), y.tolist(), alpha=0.1)

        x_test = rng.random((10, 5))
        prediction_sets = classifier.predict(x_test.tolist(), alpha=0.1)

        # with accretive, all sets should be non-empty
        assert np.all(prediction_sets.sum(axis=1) >= 1)

    def test_coverage_simulation(self, classification_model_factory: Callable[[], Any]) -> None:
        """Simulate coverage with CV+."""
        rng = np.random.default_rng(42)

        # create synthetic data
        x = rng.random((100, 5))
        y = rng.integers(0, 3, 100)

        classifier = CVPlusClassifier(
            model_factory=classification_model_factory,
            cv=10,
            random_state=42,
        )

        # split into calibration and test
        split_idx = 80
        x_cal, x_test = x[:split_idx], x[split_idx:]
        y_cal, y_test = y[:split_idx], y[split_idx:]

        # calibrate with alpha=0.1 (target coverage 90%)
        classifier.calibrate(x_cal.tolist(), y_cal.tolist(), alpha=0.1)

        # predict
        prediction_sets = classifier.predict(x_test.tolist(), alpha=0.1)

        # calculate empirical coverage
        coverage = np.mean([prediction_sets[i, y_test[i]] for i in range(len(y_test))])

        # coverage should be reasonable
        assert coverage >= 0.7, f"Coverage {coverage} too low"


def test_cvplus_inheritance() -> None:
    """Test that CVPlus inherits correctly from Jackknife."""
    # Test regressor inheritance
    assert issubclass(CVPlusRegressor, JackknifePlusRegressor)

    # Test classifier inheritance
    assert issubclass(CVPlusClassifier, JackknifePlusClassifier)

    # Test default parameters
    regressor = CVPlusRegressor(model_factory=regression_model_factory_func)
    assert regressor.cv == 5

    classifier = CVPlusClassifier(model_factory=classification_model_factory_func)
    assert classifier.cv == 5


def test_cvplus_vs_jackknife_differences() -> None:
    """Test that CV+ and Jackknife+ behave differently."""
    rng = np.random.default_rng(42)
    x = rng.random((60, 5))
    y = rng.random(60)

    def model_factory() -> MockSklearnModel:
        return MockSklearnModel(output_dim=1)

    # CV+ with 5-fold
    cvplus = CVPlusRegressor(
        model_factory=model_factory,
        cv=5,
        random_state=42,
    )

    # Jackknife+ with LOO
    jackknife = JackknifePlusRegressor(
        model_factory=model_factory,
        cv=None,  # LOO
        random_state=42,
    )

    # both should calibrate
    cvplus.calibrate(x.tolist(), y.tolist(), alpha=0.1)
    jackknife.calibrate(x.tolist(), y.tolist(), alpha=0.1)

    assert cvplus.is_calibrated
    assert jackknife.is_calibrated

    # should have different number of models
    assert len(cvplus.fitted_models) == 5
    assert len(jackknife.fitted_models) == len(x)  # LOO: one model per sample

    # predictions might differ
    x_test = rng.random((5, 5))
    cvplus_intervals = cvplus.predict(x_test.tolist(), alpha=0.1)
    jackknife_intervals = jackknife.predict(x_test.tolist(), alpha=0.1)

    assert cvplus_intervals.shape == (5, 2)
    assert jackknife_intervals.shape == (5, 2)
