"""Tests for APS common functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.scores.aps.common import APSScore, aps_score_func


class MockModel:
    """Mock model for testing."""

    def __init__(self, probs: npt.NDArray[np.floating] | None = None) -> None:
        """Initialize mock model with optional probabilities."""
        # default probabilities if none provided
        self.probs: npt.NDArray[np.floating] = probs or np.array([[0.33, 0.33, 0.33]])

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Return probabilities for testing."""
        n = len(x) if hasattr(x, "__len__") else 1
        # repeat probs for each instance
        return np.repeat(self.probs, n, axis=0)

    def __call__(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Alias for predict."""
        return self.predict(x)


def test_aps_score_func_basic() -> None:
    """Test aps_score_func with basic data."""
    probs = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
    )

    scores: npt.NDArray[np.floating] = aps_score_func(probs)

    assert scores.shape == (2, 3)
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)

    expected_first = np.array([0.5, 0.8, 1.0])
    assert np.allclose(scores[0], expected_first, atol=1e-10)


def test_apsscore_calibration() -> None:
    """Test APSScore calibration."""
    model = MockModel()
    score = APSScore(model, randomize=False, random_state=42)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    calibration_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert calibration_scores.shape == (3,)
    assert np.all(calibration_scores >= 0)
    assert np.all(calibration_scores <= 1)

    expected = np.array([0.33, 0.66, 1.0])
    assert np.allclose(calibration_scores, expected, atol=0.01)


def test_apsscore_prediction() -> None:
    """Test APSScore prediction."""
    model = MockModel()
    score = APSScore(model, randomize=False, random_state=42)

    x_test = np.array([[1, 2], [3, 4]])
    prediction_scores = score.predict_nonconformity(x_test)

    assert prediction_scores.shape == (2, 3)  # (n_samples, n_classes)
    assert np.all(prediction_scores >= 0)
    assert np.all(prediction_scores <= 1)


def test_apsscore_with_randomization() -> None:
    """Test APSScore with randomization enabled."""
    model = MockModel()
    score = APSScore(model, randomize=True, random_state=42)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    calibration_scores = score.calibration_nonconformity(x_cal, y_cal)

    # with randomization, scores might be slightly different
    assert calibration_scores.shape == (3,)
    assert np.all(calibration_scores <= 1)  # Should not exceed 1


def test_apsscore_provided_probs() -> None:
    """Test APSScore with provided probabilities."""
    model = MockModel()
    score = APSScore(model, randomize=False)

    x_test = np.array([[1, 2], [3, 4]])
    provided_probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])

    # test with provided probabilities
    scores_with_probs = score.predict_nonconformity(x_test, probs=provided_probs)

    # without provided probs, uses model's probs
    scores_without_probs = score.predict_nonconformity(x_test)

    # should differ since probs are different
    assert not np.allclose(scores_with_probs, scores_without_probs)
    assert scores_with_probs.shape == (2, 3)
    assert scores_without_probs.shape == (2, 3)
