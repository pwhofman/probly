# tests/probly/conformal_prediction/scores/lac/test_common.py
"""Tests for LAC common functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.methods.common import predict_probs
from probly.conformal_prediction.scores.lac.common import (
    LACScore,
    accretive_completion,
    lac_score_func,
)


class MockModel:
    """Mock model for testing."""

    def __init__(self, probs: npt.NDArray[np.floating] | None = None) -> None:
        """Initialize mock model with optional probabilities."""
        self.probs: npt.NDArray[np.floating] = probs or np.array([[0.33, 0.33, 0.33]])

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        n = len(x) if hasattr(x, "__len__") else 1
        return np.repeat(self.probs, n, axis=0)

    def __call__(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        return self.predict(x)


@predict_probs.register(MockModel)
def predict_probs_mock(model: MockModel, x: Sequence[Any]) -> npt.NDArray[np.floating]:
    return model.predict(x)


def test_lac_score_func_basic() -> None:
    """Test lac_score_func with basic data."""
    probs: npt.NDArray[np.floating] = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
    )

    scores: npt.NDArray[np.floating] = lac_score_func(probs)

    assert scores.shape == (2, 3)
    # LAC scores are 1 - probability
    assert np.allclose(scores, 1 - probs)
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_accretive_completion() -> None:
    """Test accretive completion functionality."""
    # Test with empty sets
    prediction_sets = np.array(
        [
            [False, False, False],  # Empty - should be completed
            [True, False, False],  # Already has one - unchanged
            [False, True, True],  # Already has two - unchanged
        ],
    )

    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],  # Class 1 has highest prob (0.8)
            [0.9, 0.05, 0.05],  # Class 0 has highest prob (0.9)
            [0.3, 0.4, 0.3],  # Class 1 has highest prob (0.4)
        ],
    )

    completed = accretive_completion(prediction_sets, probabilities)

    # First row should have class 1 added (index 1)
    assert completed[0, 1]
    assert not completed[0, 0]
    assert not completed[0, 2]

    # Second row unchanged
    assert np.array_equal(completed[1], [True, False, False])

    # Third row unchanged
    assert np.array_equal(completed[2], [False, True, True])


def test_accretive_completion_no_empty() -> None:
    """Test accretive completion when no empty sets."""
    prediction_sets = np.array(
        [
            [True, False, False],
            [False, True, False],
        ],
    )

    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
        ],
    )

    completed = accretive_completion(prediction_sets, probabilities)

    # Should be unchanged
    assert np.array_equal(completed, prediction_sets)


def test_lacscore_calibration() -> None:
    """Test LACScore calibration."""
    model = MockModel()
    score = LACScore(model)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    calibration_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert calibration_scores.shape == (3,)
    # LAC scores for true labels should be 1 - p(y|x)
    # With uniform probs 0.33, expected score is 1 - 0.33 = 0.67
    expected_score = 1 - 0.33
    assert np.allclose(calibration_scores, expected_score, atol=0.01)


def test_lacscore_prediction() -> None:
    """Test LACScore prediction."""
    model = MockModel()
    score = LACScore(model)

    x_test = np.array([[1, 2], [3, 4]])
    prediction_scores = score.predict_nonconformity(x_test)

    assert prediction_scores.shape == (2, 3)  # (n_samples, n_classes)
    # Should be 1 - probabilities
    assert np.allclose(prediction_scores, 1 - 0.33, atol=0.01)
