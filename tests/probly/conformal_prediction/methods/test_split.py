"""Tests for Split Conformal method."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from probly.conformal_prediction.methods.split import (
    SplitConformal,
    SplitConformalClassifier,
    SplitConformalRegressor,
)
from probly.conformal_prediction.scores.aps.common import APSScore
from probly.conformal_prediction.scores.lac.common import LACScore


def test_split_conformal_basic() -> None:
    """Test basic split functionality."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, 10))
    y = rng.integers(0, 3, 100)

    splitter = SplitConformal(calibration_ratio=0.3, random_state=42)
    x_train, y_train, x_cal, y_cal = splitter.split(x, y)

    assert len(x_train) == 70
    assert len(x_cal) == 30
    assert x_train.shape[1] == 10
    assert len(y_train) == 70
    assert len(y_cal) == 30


def test_split_reproducibility() -> None:
    """Test that same random_state gives same results."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal((50, 5))
    y = rng.integers(0, 2, 50)

    splitter1 = SplitConformal(random_state=42)
    splitter2 = SplitConformal(random_state=42)

    x_train1, y_train1, x_cal1, y_cal1 = splitter1.split(x, y)
    x_train2, y_train2, x_cal2, y_cal2 = splitter2.split(x, y)

    assert np.array_equal(x_train1, x_train2)
    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(x_cal1, x_cal2)
    assert np.array_equal(y_cal1, y_cal2)


def test_split_validation_checks() -> None:
    """Test input validation."""
    splitter = SplitConformal()
    rng = np.random.default_rng(42)

    # Test ratio validation
    x = rng.standard_normal((10, 3))
    y = rng.integers(0, 2, 10)

    with pytest.raises(ValueError, match="calibration_ratio must be in"):
        splitter.split(x, y, calibration_ratio=1.5)

    with pytest.raises(ValueError, match="calibration_ratio must be in"):
        splitter.split(x, y, calibration_ratio=0.0)

    # Test min samples
    x_small = rng.standard_normal((1, 3))
    y_small = rng.integers(0, 2, 1)

    with pytest.raises(ValueError, match="Need at least 2 samples"):
        splitter.split(x_small, y_small)

    # Test length mismatch
    x_mismatch = rng.standard_normal((5, 3))
    y_mismatch = rng.integers(0, 2, 3)

    with pytest.raises(ValueError, match="x and y must have the same length"):
        splitter.split(x_mismatch, y_mismatch)


def test_split_conformal_classifier_with_real_scores() -> None:
    """Test SplitConformalClassifier with real score implementations (APSScore, LACScore)."""

    class MockModel:
        def __call__(self, x: Sequence[Any]) -> np.ndarray:
            return np.zeros((len(x), 3))

    model = MockModel()

    # Test with APSScore
    aps_score = APSScore(model)
    predictor_aps = SplitConformalClassifier(model, aps_score)
    assert predictor_aps.score is aps_score
    assert predictor_aps.model is model

    # Test with LACScore
    lac_score = LACScore(model)
    predictor_lac = SplitConformalClassifier(model, lac_score, use_accretive=True)
    assert predictor_lac.score is lac_score
    assert predictor_lac.use_accretive is True


# SplitConformalClassifier Tests


class MockClassificationModel:
    """Mock model for classification testing."""

    def __call__(self, x: Sequence[Any]) -> np.ndarray:
        """Return mock probabilities for 3 classes."""
        n_samples = len(x) if hasattr(x, "__len__") else 1
        # create mock probabilities that sum to 1
        probs = np.ones((n_samples, 3), dtype=float) / 3.0
        return probs


class MockClassificationScore:
    """Mock ClassificationScore for testing."""

    def __init__(self, model: Any) -> None:  # noqa: ANN401
        """Initialize with model."""
        self.model = model

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        _y_cal: Sequence[Any],
    ) -> np.ndarray:
        """Return mock calibration scores."""
        n = len(x_cal) if hasattr(x_cal, "__len__") else 1
        # return 1D array of scores
        return np.random.default_rng(42).random(n)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: Any = None,  # noqa: ANN401, ARG002
    ) -> np.ndarray:
        """Return mock prediction scores as 2D matrix."""
        n = len(x_test) if hasattr(x_test, "__len__") else 1
        k = 3  # 3 classes
        return np.random.default_rng(42).random((n, k))


class MockRegressionModel:
    """Mock model for regression testing."""

    def __call__(self, x: Sequence[Any]) -> np.ndarray:
        """Return mock predictions."""
        n_samples = len(x) if hasattr(x, "__len__") else 1
        return np.linspace(1.0, n_samples, n_samples)


class MockRegressionScore:
    """Mock RegressionScore for testing."""

    def __init__(self, model: Any) -> None:  # noqa: ANN401
        """Initialize with model."""
        self.model = model

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        _y_cal: Sequence[Any],
    ) -> np.ndarray:
        """Return mock calibration scores."""
        n = len(x_cal) if hasattr(x_cal, "__len__") else 1
        return np.random.default_rng(42).random(n)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> np.ndarray:
        """Return mock prediction scores as 1D array."""
        n = len(x_test) if hasattr(x_test, "__len__") else 1
        return np.random.default_rng(42).random(n)

    def construct_intervals(
        self,
        y_hat: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Construct symmetric intervals."""
        n = len(y_hat)
        intervals = np.zeros((n, 2), dtype=float)
        intervals[:, 0] = y_hat - threshold
        intervals[:, 1] = y_hat + threshold
        return intervals


def test_split_conformal_classifier_initialization() -> None:
    """Test SplitConformalClassifier initialization and attributes."""
    model = MockClassificationModel()
    score = MockClassificationScore(model)

    classifier = SplitConformalClassifier(model, score, use_accretive=False)

    assert classifier.model is model
    assert classifier.score is score
    assert classifier.use_accretive is False
    assert classifier.is_calibrated is False
    assert classifier.threshold is None


def test_split_conformal_classifier_calibration() -> None:
    """Test SplitConformalClassifier calibration."""
    model = MockClassificationModel()
    score = MockClassificationScore(model)
    classifier = SplitConformalClassifier(model, score)

    # create calibration data
    x_cal = [[i, i + 1] for i in range(10)]
    y_cal = [i % 3 for i in range(10)]

    # calibrate
    threshold = classifier.calibrate(x_cal, y_cal, alpha=0.1)

    assert classifier.is_calibrated is True
    assert classifier.threshold is not None
    assert isinstance(threshold, float)
    assert threshold == classifier.threshold
    assert classifier.nonconformity_scores is not None
    assert len(classifier.nonconformity_scores) == 10


def test_split_conformal_classifier_predict_output_shape() -> None:
    """Test SplitConformalClassifier predict output shape."""
    model = MockClassificationModel()
    score = MockClassificationScore(model)
    classifier = SplitConformalClassifier(model, score)

    # calibrate first
    classifier.calibrate([[1, 2]] * 5, [0, 1, 2, 0, 1], alpha=0.1)

    # Test predict with different batch sizes
    x_test_small = [[1, 2], [3, 4]]
    predictions_small = classifier.predict(x_test_small, alpha=0.1)

    assert isinstance(predictions_small, np.ndarray)
    assert predictions_small.dtype == bool
    assert predictions_small.shape == (2, 3)  # (n_samples, n_classes)

    # Test with larger batch
    x_test_large = [[i, i + 1] for i in range(10)]
    predictions_large = classifier.predict(x_test_large, alpha=0.1)

    assert predictions_large.shape == (10, 3)
    assert predictions_large.dtype == bool


def test_split_conformal_classifier_predict_requires_calibration() -> None:
    """Test that predict raises error without calibration."""
    model = MockClassificationModel()
    score = MockClassificationScore(model)
    classifier = SplitConformalClassifier(model, score)

    with pytest.raises(RuntimeError, match="must be calibrated"):
        classifier.predict([[1, 2]], alpha=0.1)


def test_split_conformal_classifier_with_different_alphas() -> None:
    """Test SplitConformalClassifier with different alpha values."""
    model = MockClassificationModel()
    score = MockClassificationScore(model)

    x_cal = [[i, i + 1] for i in range(20)]
    y_cal = [i % 3 for i in range(20)]

    # Test with strict alpha
    classifier_strict = SplitConformalClassifier(model, score)
    threshold_strict = classifier_strict.calibrate(x_cal, y_cal, alpha=0.05)

    # Test with alpha
    classifier_permissive = SplitConformalClassifier(model, score)
    threshold_permissive = classifier_permissive.calibrate(x_cal, y_cal, alpha=0.5)

    assert isinstance(threshold_strict, float)
    assert isinstance(threshold_permissive, float)
    # more permissive alpha should generally give lower threshold
    assert threshold_strict >= threshold_permissive


def test_split_conformal_classifier_accretive_option() -> None:
    """Test SplitConformalClassifier with accretive option."""
    model = MockClassificationModel()
    score = MockClassificationScore(model)

    classifier_no_accretive = SplitConformalClassifier(model, score, use_accretive=False)
    classifier_with_accretive = SplitConformalClassifier(model, score, use_accretive=True)

    assert classifier_no_accretive.use_accretive is False
    assert classifier_with_accretive.use_accretive is True


def test_split_conformal_classifier_edge_case_single_sample_calibration() -> None:
    """Test SplitConformalClassifier calibration with single sample."""
    model = MockClassificationModel()
    score = MockClassificationScore(model)
    classifier = SplitConformalClassifier(model, score)

    x_cal = [[1, 2]]
    y_cal = [0]

    threshold = classifier.calibrate(x_cal, y_cal, alpha=0.1)

    assert classifier.is_calibrated is True
    assert isinstance(threshold, float)


def test_split_conformal_classifier_edge_case_large_batch() -> None:
    """Test SplitConformalClassifier with large batch."""
    model = MockClassificationModel()
    score = MockClassificationScore(model)
    classifier = SplitConformalClassifier(model, score)

    n_samples = 100
    x_cal = [[i, i + 1] for i in range(n_samples)]
    y_cal = [i % 3 for i in range(n_samples)]

    classifier.calibrate(x_cal, y_cal, alpha=0.1)

    x_test = [[i, i + 1] for i in range(n_samples)]
    predictions = classifier.predict(x_test, alpha=0.1)

    assert predictions.shape == (n_samples, 3)


# SplitConformalRegressor Tests


def test_split_conformal_regressor_initialization() -> None:
    """Test SplitConformalRegressor initialization and attributes."""
    model = MockRegressionModel()
    score = MockRegressionScore(model)

    regressor = SplitConformalRegressor(model, score)

    assert regressor.model is model
    assert regressor.score is score
    assert regressor.is_calibrated is False
    assert regressor.threshold is None


def test_split_conformal_regressor_calibration() -> None:
    """Test SplitConformalRegressor calibration."""
    model = MockRegressionModel()
    score = MockRegressionScore(model)
    regressor = SplitConformalRegressor(model, score)

    # create calibration data
    x_cal = [[i, i + 1] for i in range(10)]
    y_cal = [float(i) for i in range(10)]

    # calibrate
    threshold = regressor.calibrate(x_cal, y_cal, alpha=0.1)

    assert regressor.is_calibrated is True
    assert regressor.threshold is not None
    assert isinstance(threshold, float)
    assert threshold == regressor.threshold
    assert regressor.nonconformity_scores is not None
    assert len(regressor.nonconformity_scores) == 10


def test_split_conformal_regressor_predict_output_shape() -> None:
    """Test SplitConformalRegressor predict output shape."""
    model = MockRegressionModel()
    score = MockRegressionScore(model)
    regressor = SplitConformalRegressor(model, score)

    # calibrate first
    regressor.calibrate([[1, 2]] * 5, [1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.1)

    # Test predict with different batch sizes
    x_test_small = [[1, 2], [3, 4]]
    predictions_small = regressor.predict(x_test_small, alpha=0.1)

    assert isinstance(predictions_small, np.ndarray)
    assert np.issubdtype(predictions_small.dtype, np.floating)
    assert predictions_small.shape == (2, 2)  # (n_samples, 2) for [lower, upper]

    # Test with larger batch
    x_test_large = [[i, i + 1] for i in range(10)]
    predictions_large = regressor.predict(x_test_large, alpha=0.1)

    assert predictions_large.shape == (10, 2)
    # check that lower < upper for all intervals
    assert np.all(predictions_large[:, 0] < predictions_large[:, 1])


def test_split_conformal_regressor_predict_requires_calibration() -> None:
    """Test that predict raises error without calibration."""
    model = MockRegressionModel()
    score = MockRegressionScore(model)
    regressor = SplitConformalRegressor(model, score)

    with pytest.raises(RuntimeError, match="must be calibrated"):
        regressor.predict([[1, 2]], alpha=0.1)


def test_split_conformal_regressor_with_different_alphas() -> None:
    """Test SplitConformalRegressor with different alpha values."""
    model = MockRegressionModel()
    score = MockRegressionScore(model)

    x_cal = [[i, i + 1] for i in range(20)]
    y_cal = [float(i) for i in range(20)]

    # Test with strict alpha
    regressor_strict = SplitConformalRegressor(model, score)
    threshold_strict = regressor_strict.calibrate(x_cal, y_cal, alpha=0.05)

    # Test with permissive alpha
    regressor_permissive = SplitConformalRegressor(model, score)
    threshold_permissive = regressor_permissive.calibrate(x_cal, y_cal, alpha=0.5)

    assert isinstance(threshold_strict, float)
    assert isinstance(threshold_permissive, float)
    # more permissive alpha should generally give lower threshold
    assert threshold_strict >= threshold_permissive


def test_split_conformal_regressor_edge_case_single_sample_calibration() -> None:
    """Test SplitConformalRegressor calibration with single sample."""
    model = MockRegressionModel()
    score = MockRegressionScore(model)
    regressor = SplitConformalRegressor(model, score)

    x_cal = [[1, 2]]
    y_cal = [1.5]

    threshold = regressor.calibrate(x_cal, y_cal, alpha=0.1)

    assert regressor.is_calibrated is True
    assert isinstance(threshold, float)


def test_split_conformal_regressor_edge_case_large_batch() -> None:
    """Test SplitConformalRegressor with large batch."""
    model = MockRegressionModel()
    score = MockRegressionScore(model)
    regressor = SplitConformalRegressor(model, score)

    n_samples = 100
    x_cal = [[i, i + 1] for i in range(n_samples)]
    y_cal = [float(i) for i in range(n_samples)]

    regressor.calibrate(x_cal, y_cal, alpha=0.1)

    x_test = [[i, i + 1] for i in range(n_samples)]
    predictions = regressor.predict(x_test, alpha=0.1)

    assert predictions.shape == (n_samples, 2)


def test_split_conformal_classifier_vs_regressor_differences() -> None:
    """Test that Classifier and Regressor behave differently."""
    class_model = MockClassificationModel()
    class_score = MockClassificationScore(class_model)
    classifier = SplitConformalClassifier(class_model, class_score)

    reg_model = MockRegressionModel()
    reg_score = MockRegressionScore(reg_model)
    regressor = SplitConformalRegressor(reg_model, reg_score)

    # calibrate both
    classifier.calibrate([[1, 2]] * 5, [0, 1, 2, 0, 1], alpha=0.1)
    regressor.calibrate([[1, 2]] * 5, [1.0, 2.0, 3.0, 4.0, 5.0], alpha=0.1)

    x_test = [[1, 2], [3, 4]]

    # classifier returns boolean matrix
    class_preds = classifier.predict(x_test, alpha=0.1)
    assert class_preds.dtype == bool
    assert class_preds.shape == (2, 3)

    # regressor returns floating intervals
    reg_preds = regressor.predict(x_test, alpha=0.1)
    assert np.issubdtype(reg_preds.dtype, np.floating)
    assert reg_preds.shape == (2, 2)
