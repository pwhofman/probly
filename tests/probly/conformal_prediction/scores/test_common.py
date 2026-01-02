"""Tests for scores common protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.scores.common import Score


class MockScore(Score):
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
    score: Score = MockScore()

    # these should all type-check correctly
    x_cal: Sequence[Any] = [[1], [2]]
    y_cal: Sequence[Any] = [0, 1]
    x_test: Sequence[Any] = [[3], [4]]

    cal_result = score.calibration_nonconformity(x_cal, y_cal)
    pred_result = score.predict_nonconformity(x_test)

    assert isinstance(cal_result, np.ndarray)
    assert isinstance(pred_result, np.ndarray)
    assert cal_result.dtype == np.floating
    assert pred_result.dtype == np.floating


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

    class CustomScore(Score):
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
