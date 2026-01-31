"""Tests for Jackknife+ Conformal Prediction with Type Annotations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from probly.conformal_prediction.methods.jackknifeplus import (
    JackknifePlusClassifier,
    JackknifePlusRegressor,
)


class MockSklearnModel:
    """Mock sklearn-like model for Jackknife testing."""

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
            # regression: return mean of training y (simulated as 0.5)
            return np.ones(n, dtype=float) * 0.5
        # classification: return uniform probabilities
        return np.ones((n, self.output_dim), dtype=float) / self.output_dim

    def __call__(self, x: Sequence[Any] | npt.NDArray[Any]) -> npt.NDArray[np.floating]:
        """Call mock model."""
        if isinstance(x, np.ndarray):
            return self.predict(x)
        return self.predict(np.asarray(x))


@pytest.fixture
def regression_model_factory() -> Callable[[], MockSklearnModel]:
    """Factory for regression models."""
    return lambda: MockSklearnModel(output_dim=1)


@pytest.fixture
def classification_model_factory() -> Callable[[], MockSklearnModel]:
    """Factory for classification models."""
    return lambda: MockSklearnModel(output_dim=3)


@pytest.fixture
def regression_data() -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Generate regression test data."""
    rng = np.random.default_rng(42)
    x = rng.random((50, 5))
    y = rng.random(50)
    return x, y


@pytest.fixture
def classification_data() -> tuple[npt.NDArray[np.floating], npt.NDArray[np.int_]]:
    """Generate classification test data."""
    rng = np.random.default_rng(42)
    x = rng.random((50, 5))
    y = rng.integers(0, 3, 50)
    return x, y


class TestJackknifePlusRegressor:
    """Tests for JackknifePlusRegressor."""

    def test_initialization(self, regression_model_factory: Callable[[], Any]) -> None:
        """Test initialization of JackknifePlusRegressor."""
        regressor = JackknifePlusRegressor(
            model_factory=regression_model_factory,
            cv=None,  # use LeaveOneOut
            random_state=42,
        )

        assert regressor.model_factory is regression_model_factory
        assert regressor.cv is None
        assert regressor.random_state == 42
        assert not regressor.is_calibrated

    def test_calibration_loo(
        self, regression_model_factory: Callable[[], Any], regression_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test calibration with LeaveOneOut."""
        x, y = regression_data
        regressor = JackknifePlusRegressor(
            model_factory=regression_model_factory,
            cv=None,  # use LeaveOneOut
            random_state=42,
        )

        alpha = 0.1
        threshold = regressor.calibrate(x.tolist(), y.tolist(), alpha)

        assert regressor.is_calibrated
        assert threshold is not None
        assert isinstance(threshold, float)
        assert 0 <= threshold <= 1

        # should have one model per fold (LOO = n_samples folds)
        assert len(regressor.fitted_models) == len(x)
        assert regressor.n_folds_actual_ == len(x)

    def test_calibration_kfold(
        self, regression_model_factory: Callable[[], Any], regression_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test calibration with K-Fold."""
        x, y = regression_data
        regressor = JackknifePlusRegressor(
            model_factory=regression_model_factory,
            cv=5,  # 5-fold CV
            random_state=42,
        )

        alpha = 0.1
        threshold = regressor.calibrate(x.tolist(), y.tolist(), alpha)

        assert regressor.is_calibrated
        assert threshold is not None

        # should have 5 models
        assert len(regressor.fitted_models) == 5
        assert regressor.n_folds_actual_ == 5

    def test_prediction(
        self, regression_model_factory: Callable[[], Any], regression_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test prediction intervals."""
        x, y = regression_data
        regressor = JackknifePlusRegressor(
            model_factory=regression_model_factory,
            cv=5,
            random_state=42,
        )

        # calibrate
        regressor.calibrate(x.tolist(), y.tolist(), alpha=0.1)

        # predict on test data
        rng = np.random.default_rng()
        x_test = rng.random((10, 5))
        intervals = regressor.predict(x_test.tolist(), alpha=0.1)

        assert intervals.shape == (10, 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])
        assert np.issubdtype(intervals.dtype, np.floating)

    def test_prediction_without_calibration(self, regression_model_factory: Callable[[], Any]) -> None:
        """Test that predict raises error without calibration."""
        regressor = JackknifePlusRegressor(
            model_factory=regression_model_factory,
            cv=5,
        )

        with pytest.raises(RuntimeError, match="must be calibrated"):
            regressor.predict([[1, 2, 3, 4, 5]], alpha=0.1)

    def test_custom_score_func(
        self, regression_model_factory: Callable[[], Any], regression_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test with custom score function."""
        x, y = regression_data

        def custom_score(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> npt.NDArray[np.floating]:
            # absolute error
            return cast(npt.NDArray[np.floating], np.abs(y_true - y_pred))

        regressor = JackknifePlusRegressor(
            model_factory=regression_model_factory,
            cv=5,
            score_func=custom_score,
        )

        regressor.calibrate(x.tolist(), y.tolist(), alpha=0.1)
        assert regressor.is_calibrated

    def test_custom_interval_func(
        self, regression_model_factory: Callable[[], Any], regression_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test with custom interval function."""
        x, y = regression_data

        def custom_interval(
            predictions: npt.NDArray[Any], scores: npt.NDArray[Any]
        ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
            # asymmetric intervals
            # assuming predictions and scores are float-like arrays
            lower = predictions - 2 * scores
            upper = predictions + scores
            return cast(npt.NDArray[np.floating], lower), cast(npt.NDArray[np.floating], upper)

        regressor = JackknifePlusRegressor(
            model_factory=regression_model_factory,
            cv=5,
            interval_func=custom_interval,
        )

        regressor.calibrate(x.tolist(), y.tolist(), alpha=0.1)

        rng = np.random.default_rng()
        x_test = rng.random((5, 5))
        intervals = regressor.predict(x_test.tolist(), alpha=0.1)

        assert intervals.shape == (5, 2)


class TestJackknifePlusClassifier:
    """Tests for JackknifePlusClassifier."""

    def test_initialization(self, classification_model_factory: Callable[[], Any]) -> None:
        """Test initialization of JackknifePlusClassifier."""
        classifier = JackknifePlusClassifier(
            model_factory=classification_model_factory,
            cv=None,  # use LeaveOneOut
            random_state=42,
            use_accretive=False,
        )

        assert classifier.model_factory is classification_model_factory
        assert classifier.cv is None
        assert classifier.random_state == 42
        assert classifier.use_accretive is False
        assert not classifier.is_calibrated

    def test_calibration_loo(
        self, classification_model_factory: Callable[[], Any], classification_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test calibration with LeaveOneOut."""
        x, y = classification_data
        classifier = JackknifePlusClassifier(
            model_factory=classification_model_factory,
            cv=None,  # LeaveOneOut
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

    def test_prediction(
        self, classification_model_factory: Callable[[], Any], classification_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test prediction sets."""
        x, y = classification_data
        classifier = JackknifePlusClassifier(
            model_factory=classification_model_factory,
            cv=5,
            random_state=42,
        )

        # calibrate
        classifier.calibrate(x.tolist(), y.tolist(), alpha=0.1)

        # predict on test data
        rng = np.random.default_rng()
        x_test = rng.random((10, 5))
        prediction_sets = classifier.predict(x_test.tolist(), alpha=0.1)

        assert prediction_sets.shape == (10, 3)  # 10 samples, 3 classes
        assert prediction_sets.dtype == np.bool_
        assert np.all(prediction_sets.sum(axis=1) >= 1)  # At least one class per set

    def test_prediction_with_accretive(
        self, classification_model_factory: Callable[[], Any], classification_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test prediction with accretive completion."""
        x, y = classification_data
        classifier = JackknifePlusClassifier(
            model_factory=classification_model_factory,
            cv=5,
            random_state=42,
            use_accretive=True,
        )

        classifier.calibrate(x.tolist(), y.tolist(), alpha=0.1)

        rng = np.random.default_rng()
        x_test = rng.random((10, 5))
        prediction_sets = classifier.predict(x_test.tolist(), alpha=0.1)

        # with accretive, all sets should be non-empty
        assert np.all(prediction_sets.sum(axis=1) >= 1)

    def test_custom_score_func(
        self, classification_model_factory: Callable[[], Any], classification_data: tuple[npt.NDArray, npt.NDArray]
    ) -> None:
        """Test with custom score function."""
        x, y = classification_data

        def custom_score(y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> npt.NDArray[np.floating]:
            # LAC-style scores
            n = len(y_true)
            scores = np.zeros(n)
            for i in range(n):
                true_class = int(y_true[i])
                scores[i] = 1.0 - y_pred[i, true_class]
            return cast(npt.NDArray[np.floating], scores)

        classifier = JackknifePlusClassifier(
            model_factory=classification_model_factory,
            cv=5,
            score_func=custom_score,
        )

        classifier.calibrate(x.tolist(), y.tolist(), alpha=0.1)
        assert classifier.is_calibrated

    def test_coverage_guarantee_simulation(self, classification_model_factory: Callable[[], Any]) -> None:
        """Simulate coverage guarantee with synthetic data."""
        rng = np.random.default_rng(42)

        # create synthetic data where true class is always class 0
        x = rng.random((100, 5))
        y = np.zeros(100, dtype=int)

        classifier = JackknifePlusClassifier(
            model_factory=classification_model_factory,
            cv=10,  # 10-fold CV
            random_state=42,
        )

        # calibrate with alpha=0.1 (target coverage 90%)
        classifier.calibrate(x.tolist(), y.tolist(), alpha=0.1)

        # Test on new data
        x_test = rng.random((50, 5))
        y_test = np.zeros(50, dtype=int)
        prediction_sets = classifier.predict(x_test.tolist(), alpha=0.1)

        # calculate empirical coverage
        coverage = np.mean([prediction_sets[i, y_test[i]] for i in range(len(y_test))])

        # coverage should be high (not strictly guaranteed due to finite samples)
        assert coverage >= 0.8, f"Coverage {coverage} too low"

    def test_edge_case_small_dataset(self, classification_model_factory: Callable[[], Any]) -> None:
        """Test with very small dataset."""
        rng = np.random.default_rng()
        x = rng.random((5, 3))
        y = np.array([0, 1, 0, 1, 0])

        classifier = JackknifePlusClassifier(
            model_factory=classification_model_factory,
            cv=3,  # fewer folds than samples
            random_state=42,
        )

        # should still work
        classifier.calibrate(x.tolist(), y.tolist(), alpha=0.1)
        assert classifier.is_calibrated


def test_jackknife_regressor_coverage_guarantee() -> None:
    """Test coverage guarantee for Jackknife+ regressor."""
    rng = np.random.default_rng(42)

    # simple linear relationship
    x = rng.random((80, 3))
    y = x[:, 0] * 2 + x[:, 1] * 1 + rng.normal(0, 0.1, 80)

    class SimpleLinearModel:
        def __init__(self) -> None:
            self.coef_: np.ndarray = np.array([])
            self.intercept_: float = 0.0

        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            # simple linear regression
            x_with_intercept = np.c_[np.ones(len(x)), x]
            coeffs = np.linalg.lstsq(x_with_intercept, y, rcond=None)[0]
            self.intercept_ = coeffs[0]
            self.coef_ = coeffs[1:]

        def predict(self, x: np.ndarray) -> np.ndarray:
            if self.coef_.size == 0:
                error_msg = "Model not fitted"
                raise RuntimeError(error_msg)
            result: np.ndarray = self.intercept_ + np.dot(x, self.coef_)
            return result

        def __call__(self, x_seq: Sequence[Any]) -> np.ndarray:
            x_arr = np.asarray(x_seq)
            return self.predict(x_arr)

    regressor = JackknifePlusRegressor(
        model_factory=SimpleLinearModel,
        cv=10,
        random_state=42,
    )

    # calibrate
    regressor.calibrate(x.tolist(), y.tolist(), alpha=0.1)

    # Test on new data
    x_test = rng.random((20, 3))
    y_test = x_test[:, 0] * 2 + x_test[:, 1] * 1 + rng.normal(0, 0.1, 20)
    intervals = regressor.predict(x_test.tolist(), alpha=0.1)

    # calculate empirical coverage
    covered = np.sum((y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1]))
    coverage = covered / len(y_test)

    # coverage should be at least 1 - alpha = 0.9 (with some tolerance)
    assert coverage >= 0.8, f"Coverage {coverage} too low for alpha=0.1"
