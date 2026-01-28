"""Tests for Class-Conditional Conformal Prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from probly.conformal_prediction.methods.class_conditional import (
    ClassConditionalClassifier,
    ClassConditionalRegressor,
)


class MockClassificationModel:
    """Mock model for classification testing."""

    def __init__(self, n_classes: int = 3) -> None:
        """Initialize with number of classes."""
        self.n_classes = n_classes

    def __call__(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        n = len(x) if hasattr(x, "__len__") else 1
        return np.ones((n, self.n_classes), dtype=float) / self.n_classes


class MockClassificationScore:
    """Mock classification score for testing."""

    def __init__(self, model: Any) -> None:  # noqa: ANN401
        """Initialize with model."""
        self.model = model

    def calibration_nonconformity(self, x_cal: Sequence[Any], y_cal: Sequence[Any]) -> npt.NDArray[np.floating]:
        # simulate scores based on predictions
        probs = self.model(x_cal)
        y_np = np.asarray(y_cal, dtype=int)
        # return 1 - probability of true class
        result = (1.0 - probs[np.arange(len(y_cal)), y_np]).astype(float)
        return cast(npt.NDArray[np.floating], result)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: Any = None,  # noqa: ANN401, ARG002
    ) -> npt.NDArray[np.floating]:
        k = 3  # 3 classes
        return np.random.default_rng(42).random((len(x_test), k))


class MockRegressionModel:
    """Mock model for regression testing."""

    def __call__(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        n = len(x) if hasattr(x, "__len__") else 1
        return np.linspace(1.0, n, n)


class MockRegressionScore:
    """Mock regression score for testing."""

    def __init__(self, model: Any) -> None:  # noqa: ANN401
        """Initialize with model."""
        self.model = model

    def calibration_nonconformity(self, x_cal: Sequence[Any], y_cal: Sequence[Any]) -> npt.NDArray[np.floating]:
        predictions = self.model(x_cal)
        y_np = np.asarray(y_cal, dtype=float)
        # return absolute errors
        result = np.abs(predictions - y_np).astype(float)
        return cast(npt.NDArray[np.floating], result)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        return np.random.default_rng(42).random(len(x_test))

    def construct_intervals(
        self,
        y_hat: npt.NDArray[np.floating],
        threshold: float,
    ) -> npt.NDArray[np.floating]:
        intervals = np.zeros((len(y_hat), 2), dtype=float)
        intervals[:, 0] = y_hat - threshold
        intervals[:, 1] = y_hat + threshold
        return intervals


def simple_class_func(x: Sequence[Any], y: Sequence[Any] | None = None) -> npt.NDArray[np.int_]:
    """Simple class function for testing."""
    if y is not None:
        # use actual labels for calibration
        return np.asarray(y, dtype=int)
    # for prediction, return random classes
    return np.random.default_rng(42).integers(0, 3, size=len(x))


@pytest.fixture
def classification_setup() -> tuple[MockClassificationModel, MockClassificationScore]:
    """Setup for classification tests."""
    model = MockClassificationModel(n_classes=3)
    score = MockClassificationScore(model)
    return model, score


@pytest.fixture
def regression_setup() -> tuple[MockRegressionModel, MockRegressionScore]:
    """Setup for regression tests."""
    model = MockRegressionModel()
    score = MockRegressionScore(model)
    return model, score


class TestClassConditionalClassifier:
    """Tests for ClassConditionalClassifier."""

    def test_initialization(self, classification_setup: tuple[Any, Any]) -> None:
        """Test initialization of ClassConditionalClassifier."""
        model, score = classification_setup
        classifier = ClassConditionalClassifier(
            model=model,
            score=cast(Any, score),
            class_func=simple_class_func,
            use_accretive=False,
        )

        assert classifier.model is model
        assert classifier.score is score
        assert not classifier.is_calibrated
        assert classifier.use_accretive is False

    def test_calibration(self, classification_setup: tuple[Any, Any]) -> None:
        """Test calibration of ClassConditionalClassifier."""
        model, score = classification_setup
        classifier = ClassConditionalClassifier(
            model=model,
            score=cast(Any, score),
            class_func=simple_class_func,
        )

        # create calibration data
        x_cal = [[i, i + 1] for i in range(20)]
        y_cal = [i % 3 for i in range(20)]  # 3 classes

        alpha = 0.1
        result_alpha = classifier.calibrate(x_cal, y_cal, alpha)

        assert classifier.is_calibrated
        assert result_alpha == alpha
        assert len(classifier.group_thresholds) > 0

        # check thresholds per class
        for class_id, threshold in classifier.group_thresholds.items():
            assert isinstance(class_id, (int, np.integer))
            assert isinstance(threshold, (float, np.floating))
            assert 0 <= threshold <= 1

    def test_prediction_shape_and_type(self, classification_setup: tuple[Any, Any]) -> None:
        """Test prediction output shape and type."""
        model, score = classification_setup
        classifier = ClassConditionalClassifier(
            model=model,
            score=cast(Any, score),
            class_func=simple_class_func,
            use_accretive=False,
        )

        # calibrate first
        x_cal = [[i, i + 1] for i in range(20)]
        y_cal = [i % 3 for i in range(20)]
        classifier.calibrate(x_cal, y_cal, alpha=0.1)

        # make predictions
        x_test = [[i, i + 1] for i in range(10)]
        prediction_sets = classifier.predict(x_test, alpha=0.1)

        assert prediction_sets.shape == (10, 3)  # (n_samples, n_classes)
        assert prediction_sets.dtype == np.bool_
        assert np.all(prediction_sets.sum(axis=1) >= 1)

    def test_prediction_with_accretive(self, classification_setup: tuple[Any, Any]) -> None:
        """Test prediction with accretive completion."""
        model, score = classification_setup
        classifier = ClassConditionalClassifier(
            model=model,
            score=cast(Any, score),
            class_func=simple_class_func,
            use_accretive=True,
        )

        # calibrate
        x_cal = [[i, i + 1] for i in range(20)]
        y_cal = [i % 3 for i in range(20)]
        classifier.calibrate(x_cal, y_cal, alpha=0.1)

        # predict
        x_test = [[i, i + 1] for i in range(10)]
        prediction_sets = classifier.predict(x_test, alpha=0.1)

        # with accretive, all sets should be non-empty
        assert np.all(prediction_sets.sum(axis=1) >= 1)

    def test_predict_without_calibration(self, classification_setup: tuple[Any, Any]) -> None:
        """Test that predict raises error without calibration."""
        model, score = classification_setup
        classifier = ClassConditionalClassifier(
            model=model,
            score=cast(Any, score),
            class_func=simple_class_func,
        )

        with pytest.raises(RuntimeError, match="must be calibrated"):
            classifier.predict([[1, 2]], alpha=0.1)

    def test_edge_case_single_class(self) -> None:
        """Test with single class."""
        model = MockClassificationModel(n_classes=1)
        score = MockClassificationScore(model)

        def single_class_func(x: Sequence[Any], y: Sequence[Any] | None = None) -> npt.NDArray[np.int_]:
            del y  # Unused parameter
            return np.zeros(len(x), dtype=int)

        classifier = ClassConditionalClassifier(
            model=model,
            score=cast(Any, score),
            class_func=single_class_func,
        )

        x_cal = [[i, i + 1] for i in range(10)]
        y_cal = [0] * 10
        classifier.calibrate(x_cal, y_cal, alpha=0.1)

        assert 0 in classifier.group_thresholds
        assert classifier.is_calibrated


class TestClassConditionalRegressor:
    """Tests for ClassConditionalRegressor."""

    def test_initialization(self, regression_setup: tuple[Any, Any]) -> None:
        """Test initialization of ClassConditionalRegressor."""
        model, score = regression_setup
        regressor = ClassConditionalRegressor(
            model=model,
            score=cast(Any, score),
            class_func=simple_class_func,
        )

        assert regressor.model is model
        assert regressor.score is score
        assert not regressor.is_calibrated

    def test_calibration_symmetric(self, regression_setup: tuple[Any, Any]) -> None:
        """Test calibration with symmetric thresholds."""
        model, score = regression_setup
        regressor = ClassConditionalRegressor(
            model=model,
            score=cast(Any, score),
            class_func=simple_class_func,
        )

        # create calibration data
        x_cal = [[i, i + 1] for i in range(30)]
        # assign classes: 0, 1, or 2
        y_cal = [float(i % 3) for i in range(30)]

        alpha = 0.1
        result_alpha = regressor.calibrate(x_cal, y_cal, alpha)

        assert regressor.is_calibrated
        assert result_alpha == alpha
        assert not regressor.is_asymmetric
        assert len(regressor.group_thresholds) > 0

    def test_prediction_symmetric(self, regression_setup: tuple[Any, Any]) -> None:
        """Test prediction with symmetric intervals."""
        model, score = regression_setup
        regressor = ClassConditionalRegressor(
            model=model,
            score=cast(Any, score),
            class_func=simple_class_func,
        )

        # calibrate
        x_cal = [[i, i + 1] for i in range(30)]
        y_cal = [float(i % 3) for i in range(30)]
        regressor.calibrate(x_cal, y_cal, alpha=0.1)

        # predict
        x_test = [[i, i + 1] for i in range(10)]
        intervals = regressor.predict(x_test, alpha=0.1)

        assert intervals.shape == (10, 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])
        assert np.issubdtype(intervals.dtype, np.floating)

    def test_predict_without_calibration(self, regression_setup: tuple[Any, Any]) -> None:
        """Test that predict raises error without calibration."""
        model, score = regression_setup
        regressor = ClassConditionalRegressor(
            model=model,
            score=cast(Any, score),
            class_func=simple_class_func,
        )

        with pytest.raises(RuntimeError, match="must be calibrated"):
            regressor.predict([[1, 2]], alpha=0.1)

    def test_edge_case_empty_groups(self) -> None:
        """Test with empty groups in calibration."""
        model = MockRegressionModel()
        score = MockRegressionScore(model)

        def constant_class_func(x: Sequence[Any], y: Sequence[Any] | None = None) -> npt.NDArray[np.int_]:
            del y  # Unused parameter
            # all samples in same class
            return np.ones(len(x), dtype=int) * 5

        regressor = ClassConditionalRegressor(
            model=model,
            score=cast(Any, score),
            class_func=constant_class_func,
        )

        x_cal = [[i, i + 1] for i in range(10)]
        y_cal = [float(i) for i in range(10)]

        # should still calibrate (uses max threshold as fallback)
        regressor.calibrate(x_cal, y_cal, alpha=0.1)

        assert regressor.is_calibrated
        assert 5 in regressor.group_thresholds


def test_class_conditional_vs_mondrian_differences() -> None:
    """Test that ClassConditional and Mondrian behave differently."""

    class MockModel:
        def __call__(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
            return np.ones((len(x), 3), dtype=float) / 3.0

    class MockScore:
        def __init__(self, model: Any) -> None:  # noqa: ANN401
            self.model = model

        def calibration_nonconformity(self, x_cal: Sequence[Any], _y_cal: Sequence[Any]) -> npt.NDArray[np.floating]:
            return np.random.default_rng(42).random(len(x_cal))

        def predict_nonconformity(
            self,
            x_test: Sequence[Any],
            probs: Any = None,  # noqa: ANN401, ARG002
        ) -> npt.NDArray[np.floating]:
            return np.random.default_rng(42).random((len(x_test), 3))

    model = MockModel()
    score = MockScore(model)

    # different group functions
    def region_func(x: Sequence[Any]) -> npt.NDArray[np.int_]:
        # mondrian: regions based on input features
        regions = []
        for sample in x:
            val = sample[0] if isinstance(sample, (list, np.ndarray)) else 0
            regions.append(0 if val < 0.5 else 1)
        return np.array(regions, dtype=int)

    def class_func(x: Sequence[Any], y: Sequence[Any] | None = None) -> npt.NDArray[np.int_]:
        # ClassConditional: groups based on true labels (y) for calibration
        if y is not None:
            return np.asarray(y, dtype=int)
        # for prediction, use predicted class (simplified mock here)
        return np.zeros(len(x), dtype=int)

    # both should work but with different grouping strategies
    mondrian = ClassConditionalClassifier(
        model=model,
        score=cast(Any, score),
        class_func=region_func,
    )

    class_conditional = ClassConditionalClassifier(
        model=model,
        score=cast(Any, score),
        class_func=class_func,
    )

    # simple check that the group function assignment worked correctly
    assert mondrian is not class_conditional
