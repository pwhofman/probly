"""Tests for scores common protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.scores.common import (
    ClassificationScore,
    RegressionScore,
)


class MockScore(ClassificationScore):
    """Mock implementation of Score protocol for testing."""

    def __init__(self, fixed_score: float = 0.5) -> None:
        """Initialize mock score with fixed value."""
        self.fixed_score = fixed_score
        super().__init__(model=lambda _: None, score_func=lambda _: np.ones((len(_), 3), dtype=float) * fixed_score)

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        _y_cal: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Return fixed scores for calibration."""
        n = len(x_cal) if hasattr(x_cal, "__len__") else 1
        return np.ones(n, dtype=float) * self.fixed_score

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: Any = None,  # noqa: ARG002, ANN401
    ) -> npt.NDArray[np.floating]:
        """Return fixed score matrix for prediction."""
        n = len(x_test) if hasattr(x_test, "__len__") else 1
        k = 3  # 3 classes
        return np.ones((n, k), dtype=float) * self.fixed_score


class MockRegressionScore(RegressionScore):
    """Mock implementation of RegressionScore protocol for testing."""

    def __init__(self, fixed_score: float = 0.5) -> None:
        """Initialize mock regression score with fixed value."""
        self.fixed_score = fixed_score
        super().__init__(model=lambda _: None, score_func=lambda y, _: np.ones(len(y), dtype=float) * fixed_score)

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        _y_cal: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Return fixed scores for calibration."""
        n = len(x_cal) if hasattr(x_cal, "__len__") else 1
        return np.ones(n, dtype=float) * self.fixed_score

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Return fixed 1D scores for prediction."""
        n = len(x_test) if hasattr(x_test, "__len__") else 1
        return np.ones(n, dtype=float) * self.fixed_score

    def construct_intervals(
        self,
        y_hat: npt.NDArray[np.floating],
        threshold: float,
    ) -> npt.NDArray[np.floating]:
        """Construct symmetric intervals around predictions."""
        n = len(y_hat)
        intervals = np.zeros((n, 2), dtype=float)
        intervals[:, 0] = y_hat - threshold  # lower bound
        intervals[:, 1] = y_hat + threshold  # upper bound
        return intervals


def test_output_types() -> None:
    """Test that MockScore outputs correct types."""
    score = MockScore()

    x_cal = [[1], [2], [3]]
    y_cal = [0, 1, 0]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert np.issubdtype(cal_scores.dtype, np.floating)

    x_test = [[4], [5]]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert np.issubdtype(pred_scores.dtype, np.floating)


def test_edge_cases_single_sample() -> None:
    """Test MockScore with single sample inputs."""
    score = MockScore(fixed_score=0.1)

    x_cal = [[1, 2]]
    y_cal = [0]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.shape == (1,)
    assert np.all(cal_scores == 0.1)

    x_test = [[3, 4]]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert pred_scores.shape == (1, 3)  # 1 sample, 3 classes
    assert np.all(pred_scores == 0.1)


def test_edge_cases_large_batch() -> None:
    """Test MockScore with large batch inputs."""
    score = MockScore(fixed_score=0.7)

    n_samples = 1000
    x_cal = [[i, i + 1] for i in range(n_samples)]
    y_cal = [i % 3 for i in range(n_samples)]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.shape == (n_samples,)
    assert np.all(cal_scores == 0.7)

    x_test = [[i, i + 1] for i in range(n_samples)]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert pred_scores.shape == (n_samples, 3)  # n_samples, 3 classes
    assert np.all(pred_scores == 0.7)


def test_no_input_modification() -> None:
    """Test that MockScore does not modify input data."""
    score = MockScore(fixed_score=0.6)

    x_cal = [[1, 2], [3, 4]]
    y_cal = [0, 1]
    x_cal_copy = [list(x) for x in x_cal]
    y_cal_copy = list(y_cal)

    _ = score.calibration_nonconformity(x_cal, y_cal)

    assert x_cal == x_cal_copy
    assert y_cal == y_cal_copy

    x_test = [[5, 6], [7, 8]]
    x_test_copy = [list(x) for x in x_test]

    _ = score.predict_nonconformity(x_test)

    assert x_test == x_test_copy


def test_forward_shape() -> None:
    """Test that MockScore returns correct output shapes for various batch sizes."""
    score = MockScore(fixed_score=0.3)

    # Test calibration shapes with small batch
    cal_scores_small = score.calibration_nonconformity([[1, 2], [3, 4]], [0, 1])
    assert isinstance(cal_scores_small, np.ndarray)
    assert cal_scores_small.shape == (2,)  # 1D array

    # Test calibration with medium batch
    cal_scores_medium = score.calibration_nonconformity([[1, 2]] * 5, [0] * 5)
    assert cal_scores_medium.shape == (5,)

    # Test calibration with larger batch
    cal_scores_large = score.calibration_nonconformity([[1, 2]] * 10, [0] * 10)
    assert cal_scores_large.shape == (10,)

    # Test prediction shapes with different batch sizes
    pred_scores_small = score.predict_nonconformity([[5, 6], [7, 8]])
    assert isinstance(pred_scores_small, np.ndarray)
    assert pred_scores_small.shape == (2, 3)  # 2D: (n_samples, n_classes)

    pred_scores_medium = score.predict_nonconformity([[1, 2]] * 5)
    assert pred_scores_medium.shape == (5, 3)

    pred_scores_large = score.predict_nonconformity([[1, 2]] * 10)
    assert pred_scores_large.shape == (10, 3)


def test_score_protocol_implementation() -> None:
    """Test that ClassificationScore protocol is correctly implemented."""
    score = MockScore(fixed_score=0.5)

    # Test that protocol methods exist and are callable
    assert hasattr(score, "calibration_nonconformity")
    assert hasattr(score, "predict_nonconformity")
    assert callable(score.calibration_nonconformity)
    assert callable(score.predict_nonconformity)

    # Test calibration method returns correct type
    x_cal = [[1, 2], [3, 4]]
    y_cal = [0, 1]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert np.issubdtype(cal_scores.dtype, np.floating)

    # Test prediction method returns correct type
    x_test = [[5, 6], [7, 8]]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert np.issubdtype(pred_scores.dtype, np.floating)

    # Test that score can be used as ClassificationScore type annotation
    score_typed: ClassificationScore = MockScore()
    assert hasattr(score_typed, "calibration_nonconformity")
    assert hasattr(score_typed, "predict_nonconformity")


def test_score_protocol_with_probs() -> None:
    """Test Score protocol with provided probabilities."""
    score = MockScore(fixed_score=0.4)

    x_test = [[1, 2]]
    probs = np.array([[0.8, 0.1, 0.1]])

    # should work with or without probs
    scores_with_probs = score.predict_nonconformity(x_test, probs=probs)
    scores_without_probs = score.predict_nonconformity(x_test)

    assert scores_with_probs.shape == (1, 3)
    assert scores_without_probs.shape == (1, 3)
    # our mock ignores probs, so both should be same
    assert np.array_equal(scores_with_probs, scores_without_probs)


def test_score_protocol_type_hints() -> None:
    """Test that Score protocol has correct type hints."""
    # this is more of a type checking test
    score: ClassificationScore = MockScore()

    # these should all type-check correctly
    x_cal: Sequence[Any] = [[1], [2]]
    y_cal: Sequence[Any] = [0, 1]
    x_test: Sequence[Any] = [[3], [4]]

    cal_result = score.calibration_nonconformity(x_cal, y_cal)
    pred_result = score.predict_nonconformity(x_test)

    assert isinstance(cal_result, np.ndarray)
    assert isinstance(pred_result, np.ndarray)
    assert np.issubdtype(cal_result.dtype, np.floating)
    assert np.issubdtype(pred_result.dtype, np.floating)


def test_score_inheritance() -> None:
    """Test that classes can properly inherit from Score."""

    class CustomScore(ClassificationScore):
        """Custom score implementation."""

        def calibration_nonconformity(
            self,
            _x_cal: Sequence[Any],
            _y_cal: Sequence[Any],
        ) -> npt.NDArray[np.floating]:
            """Custom calibration."""
            return np.array([1.0, 2.0, 3.0], dtype=float)

        def predict_nonconformity(
            self,
            x_test: Sequence[Any],
            probs: Any = None,  # noqa: ARG002, ANN401
        ) -> npt.NDArray[np.floating]:
            """Custom prediction."""
            n = len(x_test) if hasattr(x_test, "__len__") else 1
            return np.ones((n, 4), dtype=float) * 0.5

    score = CustomScore()

    # should implement all required methods
    cal_scores = score.calibration_nonconformity([], [])
    assert cal_scores.shape == (3,)

    pred_scores = score.predict_nonconformity([[1], [2], [3]])
    assert pred_scores.shape == (3, 4)


# Regression Score Tests


def test_regression_forward_shape() -> None:
    """Test that MockRegressionScore returns correct output shapes."""
    score = MockRegressionScore(fixed_score=0.3)

    # Test calibration shapes with different batch sizes
    x_cal = [[1, 2], [3, 4]]
    y_cal = [0.5, 1.5]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.shape == (2,)  # 1D for regression

    # Test with larger calibration set
    x_cal_large = [[1, 2], [3, 4], [5, 6]]
    y_cal_large = [0.5, 1.5, 2.5]
    cal_scores_large = score.calibration_nonconformity(x_cal_large, y_cal_large)

    assert cal_scores_large.shape == (3,)

    # Test prediction shapes (1D for regression, not 2D like classification)
    x_test = [[5, 6], [7, 8], [9, 10]]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert pred_scores.shape == (3,)  # 1D, not (3, n_classes)

    # Test with smaller prediction set
    x_test_small = [[7, 8], [9, 10]]
    pred_scores_small = score.predict_nonconformity(x_test_small)

    assert pred_scores_small.shape == (2,)


def test_regression_output_types() -> None:
    """Test that MockRegressionScore outputs correct types."""
    score = MockRegressionScore()

    x_cal = [[1], [2], [3]]
    y_cal = [0.5, 1.0, 1.5]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert np.issubdtype(cal_scores.dtype, np.floating)

    x_test = [[4], [5]]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert np.issubdtype(pred_scores.dtype, np.floating)


def test_regression_construct_intervals() -> None:
    """Test that construct_intervals works correctly."""
    score = MockRegressionScore(fixed_score=0.2)

    # Test with simple predictions
    y_hat = np.array([1.0, 2.0, 3.0])
    threshold = 0.5

    intervals = score.construct_intervals(y_hat, threshold)

    assert isinstance(intervals, np.ndarray)
    assert intervals.shape == (3, 2)  # (n_samples, 2) for [lower, upper]

    # check that intervals are symmetric around predictions
    expected_lower = y_hat - threshold
    expected_upper = y_hat + threshold

    np.testing.assert_array_almost_equal(intervals[:, 0], expected_lower)
    np.testing.assert_array_almost_equal(intervals[:, 1], expected_upper)

    # Test with different threshold
    threshold_large = 2.0
    intervals_large = score.construct_intervals(y_hat, threshold_large)

    assert intervals_large.shape == (3, 2)
    np.testing.assert_array_almost_equal(intervals_large[:, 0], y_hat - threshold_large)
    np.testing.assert_array_almost_equal(intervals_large[:, 1], y_hat + threshold_large)


def test_regression_construct_intervals_single_sample() -> None:
    """Test construct_intervals with single sample."""
    score = MockRegressionScore()

    y_hat = np.array([5.0])
    threshold = 1.0

    intervals = score.construct_intervals(y_hat, threshold)

    assert intervals.shape == (1, 2)
    assert intervals[0, 0] == 4.0  # lower bound
    assert intervals[0, 1] == 6.0  # upper bound


def test_regression_edge_cases_single_sample() -> None:
    """Test MockRegressionScore with single sample inputs."""
    score = MockRegressionScore(fixed_score=0.1)

    x_cal = [[1, 2]]
    y_cal = [0.5]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.shape == (1,)
    assert np.all(cal_scores == 0.1)

    x_test = [[3, 4]]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert pred_scores.shape == (1,)  # 1D for regression
    assert np.all(pred_scores == 0.1)


def test_regression_edge_cases_large_batch() -> None:
    """Test MockRegressionScore with large batch inputs."""
    score = MockRegressionScore(fixed_score=0.7)

    n_samples = 1000
    x_cal = [[i, i + 1] for i in range(n_samples)]
    y_cal = [float(i) for i in range(n_samples)]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.shape == (n_samples,)
    assert np.all(cal_scores == 0.7)

    x_test = [[i, i + 1] for i in range(n_samples)]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert pred_scores.shape == (n_samples,)  # 1D for regression
    assert np.all(pred_scores == 0.7)


def test_regression_score_protocol_implementation() -> None:
    """Test that RegressionScore protocol is correctly implemented."""
    score = MockRegressionScore(fixed_score=0.5)

    # Test that protocol methods exist and are callable
    assert hasattr(score, "calibration_nonconformity")
    assert hasattr(score, "predict_nonconformity")
    assert hasattr(score, "construct_intervals")
    assert callable(score.calibration_nonconformity)
    assert callable(score.predict_nonconformity)
    assert callable(score.construct_intervals)

    # Test calibration method returns correct type
    x_cal = [[1, 2], [3, 4]]
    y_cal = [0.5, 1.5]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert np.issubdtype(cal_scores.dtype, np.floating)

    # Test prediction method returns correct type and shape
    x_test = [[5, 6], [7, 8]]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert np.issubdtype(pred_scores.dtype, np.floating)
    assert pred_scores.ndim == 1  # Must be 1D for regression

    # Test construct_intervals returns correct type and shape
    y_hat = np.array([1.0, 2.0])
    intervals = score.construct_intervals(y_hat, threshold=0.5)

    assert isinstance(intervals, np.ndarray)
    assert intervals.shape == (2, 2)  # (n_samples, 2)
    assert np.issubdtype(intervals.dtype, np.floating)

    # Test that score can be used as RegressionScore type annotation
    score_typed: RegressionScore = MockRegressionScore()
    assert hasattr(score_typed, "calibration_nonconformity")
    assert hasattr(score_typed, "predict_nonconformity")
    assert hasattr(score_typed, "construct_intervals")


def test_regression_no_input_modification() -> None:
    """Test that MockRegressionScore does not modify input data."""
    score = MockRegressionScore(fixed_score=0.6)

    x_cal = [[1, 2], [3, 4]]
    y_cal = [0.5, 1.5]
    x_cal_copy = [list(x) for x in x_cal]
    y_cal_copy = list(y_cal)

    _ = score.calibration_nonconformity(x_cal, y_cal)

    assert x_cal == x_cal_copy
    assert y_cal == y_cal_copy

    x_test = [[5, 6], [7, 8]]
    x_test_copy = [list(x) for x in x_test]

    _ = score.predict_nonconformity(x_test)

    assert x_test == x_test_copy

    # Test construct_intervals doesn't modify y_hat
    y_hat = np.array([1.0, 2.0, 3.0])
    y_hat_copy = y_hat.copy()

    _ = score.construct_intervals(y_hat, threshold=0.5)

    np.testing.assert_array_equal(y_hat, y_hat_copy)


def test_regression_score_inheritance() -> None:
    """Test that classes can properly inherit from RegressionScore."""

    class CustomRegressionScore(RegressionScore):
        """Custom regression score implementation."""

        def calibration_nonconformity(
            self,
            _x_cal: Sequence[Any],
            _y_cal: Sequence[Any],
        ) -> npt.NDArray[np.floating]:
            """Custom calibration."""
            return np.array([1.0, 2.0, 3.0], dtype=float)

        def predict_nonconformity(
            self,
            x_test: Sequence[Any],
        ) -> npt.NDArray[np.floating]:
            """Custom prediction."""
            n = len(x_test) if hasattr(x_test, "__len__") else 1
            return np.ones(n, dtype=float) * 0.5

        def construct_intervals(
            self,
            y_hat: npt.NDArray[np.floating],
            threshold: float,
        ) -> npt.NDArray[np.floating]:
            """Custom interval construction."""
            n = len(y_hat)
            intervals = np.zeros((n, 2), dtype=float)
            intervals[:, 0] = y_hat - 2 * threshold  # asymmetric
            intervals[:, 1] = y_hat + threshold
            return intervals

    score = CustomRegressionScore()

    # should implement all required methods
    cal_scores = score.calibration_nonconformity([], [])
    assert cal_scores.shape == (3,)

    pred_scores = score.predict_nonconformity([[1], [2], [3]])
    assert pred_scores.shape == (3,)

    # Test custom interval construction
    y_hat = np.array([5.0])
    intervals = score.construct_intervals(y_hat, threshold=1.0)
    assert intervals.shape == (1, 2)
    assert intervals[0, 0] == 3.0  # y_hat - 2*threshold
    assert intervals[0, 1] == 6.0  # y_hat + threshold
