"""Tests for scores common protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.scores.common import ClassificationScore


class MockScore(ClassificationScore):
    """Mock implementation of Score protocol for testing."""

    def __init__(self, fixed_score: float = 0.5) -> None:
        """Initialize mock score with fixed value."""
        self.fixed_score = fixed_score

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


def test_forward_shape() -> None:
    """Test that MockScore returns correct shapes."""
    score = MockScore(fixed_score=0.2)

    x_cal = [[1, 2], [3, 4], [5, 6]]
    y_cal = [0, 1, 2]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.shape == (3,)
    assert np.all(cal_scores == 0.2)

    x_test = [[7, 8], [9, 10]]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert pred_scores.shape == (2, 3)  # 2 samples, 3 classes
    assert np.all(pred_scores == 0.2)


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


def test_score_protocol_implementation() -> None:
    """Test that a class implementing Score protocol works correctly."""
    score = MockScore(fixed_score=0.3)

    # Test calibration method
    x_cal = [[1, 2], [3, 4]]
    y_cal = [0, 1]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.shape == (2,)
    assert np.all(cal_scores == 0.3)

    # Test prediction method
    x_test = [[5, 6], [7, 8], [9, 10]]
    pred_scores = score.predict_nonconformity(x_test)

    assert isinstance(pred_scores, np.ndarray)
    assert pred_scores.shape == (3, 3)  # 3 samples, 3 classes
    assert np.all(pred_scores == 0.3)


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


def test_score_protocol_output_shapes() -> None:
    """Test Score protocol expected output shapes."""
    score = MockScore()

    # Calibration should return 1D array
    cal_scores = score.calibration_nonconformity([[1, 2]] * 5, [0] * 5)
    assert cal_scores.shape == (5,)

    # Prediction should return 2D array
    pred_scores = score.predict_nonconformity([[1, 2]] * 10)
    assert pred_scores.shape == (10, 3)


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
