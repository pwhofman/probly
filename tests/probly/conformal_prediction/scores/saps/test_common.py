"""Tests for SAPS common functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from probly.conformal_prediction.scores.saps.common import SAPSScore, saps_score_func


class MockModel:
    """Mock model for testing."""

    def __init__(self, probs: npt.NDArray[np.floating] | None = None, n_classes: int = 3) -> None:
        """Initialize mock model."""
        if probs is not None:
            self.probs: npt.NDArray[np.floating] = probs if probs.ndim == 2 else probs.reshape(1, -1)
        else:
            # default uniform probabilities
            self.probs = np.ones((1, n_classes)) / n_classes

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Return probabilities for testing."""
        n = len(x) if hasattr(x, "__len__") else 1
        # repeat probs for each instance
        return np.repeat(self.probs, n, axis=0)

    def __call__(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Alias for predict."""
        return self.predict(x)


def test_saps_score_func_basic() -> None:
    """Test saps_score_func with basic data."""
    probs = np.array([0.5, 0.3, 0.2], dtype=float)
    label = 0  # class with highest prob

    # test with rank 1
    score: float = saps_score_func(probs, label, lambda_val=0.1, u=0.5)

    assert isinstance(score, float)

    assert np.isclose(score, 0.25, atol=1e-10)

    # test with rank >= 1
    label = 1  # class with second highest prob
    score = saps_score_func(probs, label, lambda_val=0.1, u=0.5)

    # for rank 2: max_prob + (rank -2 + u) * lambda_val
    assert np.isclose(score, 0.55, atol=1e-10)


def test_saps_score_func_invalid_label() -> None:
    """Test saps_score_func with invalid label."""
    probs = np.array([0.5, 0.3, 0.2], dtype=float)

    with pytest.raises(ValueError, match=".*"):
        saps_score_func(probs, label=-1, lambda_val=0.1)

    with pytest.raises(ValueError, match=".*"):
        saps_score_func(probs, label=3, lambda_val=0.1)


def test_saps_score_func_batch_basic() -> None:
    """Test saps_score_func_batch with basic data."""
    probs = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
        dtype=float,
    )
    labels = np.array([0, 1], dtype=int)
    us = np.array([0.5, 0.5], dtype=float)

    scores = saps_score_func(probs, labels, lambda_val=0.1, us=us)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert scores.dtype in (np.float32, np.float64)

    expected = np.array([0.25, 0.35], dtype=float)
    assert np.allclose(scores, expected, atol=1e-10)


def test_saps_score_func_batch_without_us() -> None:
    """Test saps_score_func_batch without providing us."""
    probs = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
        dtype=float,
    )
    labels = np.array([0, 1], dtype=int)

    scores = saps_score_func(probs, labels, lambda_val=0.1, us=None)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert scores.dtype in [np.float32, np.float64]
    # Should be in valid range [0, max_prob + (K-1)*lambda]
    assert np.all(scores >= 0)
    max_probs = np.max(probs, axis=1)
    assert np.all(scores <= max_probs + (probs.shape[1] - 1) * 0.1 + 1e-6)


def test_sapsscore_calibration() -> None:
    """Test SAPSScore calibration."""
    model = MockModel()
    score = SAPSScore(model, lambda_val=0.1, random_state=42)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    calibration_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert calibration_scores.shape == (3,)
    assert np.all(calibration_scores >= 0)
    assert isinstance(calibration_scores, np.ndarray)
    assert calibration_scores.dtype in (np.float32, np.float64)


def test_sapsscore_prediction() -> None:
    """Test SAPSScore prediction."""
    model = MockModel(probs=np.array([[0.8, 0.1, 0.1]]))
    score = SAPSScore(model, lambda_val=0.1, random_state=42)

    x_test = np.array([[1, 2], [3, 4]])
    prediction_scores = score.predict_nonconformity(x_test)

    assert prediction_scores.shape == (2, 3)  # (n_samples, n_classes)
    assert np.all(prediction_scores >= 0)
    assert isinstance(prediction_scores, np.ndarray)
    assert prediction_scores.dtype in (np.float32, np.float64)


def test_sapsscore_with_different_lambda() -> None:
    """Test SAPSScore with different lambda values."""
    model = MockModel(probs=np.array([[0.5, 0.3, 0.2]]))

    score_small_lambda = SAPSScore(model, lambda_val=0.01, random_state=42)
    score_large_lambda = SAPSScore(model, lambda_val=0.5, random_state=42)

    x_cal = np.array([[1, 2]])
    y_cal = np.array([1])  # rank 2

    scores_small_lambda = score_small_lambda.calibration_nonconformity(x_cal, y_cal)
    scores_large_lambda = score_large_lambda.calibration_nonconformity(x_cal, y_cal)

    assert scores_large_lambda[0] > scores_small_lambda[0]


def test_sapsscore_provided_probs() -> None:
    """Test SAPSScore with provided probabilities."""
    model = MockModel(probs=np.array([[0.33, 0.33, 0.34]]))
    score = SAPSScore(model, lambda_val=0.1, random_state=42)

    x_test = np.array([[1, 2], [3, 4]])
    provided_probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
        ],
    )

    # test with provided probs
    scores_with_probs = score.predict_nonconformity(x_test, probs=provided_probs)

    # without provided probs, uses model's probs
    scores_without_probs = score.predict_nonconformity(x_test)

    assert not np.allclose(scores_with_probs, scores_without_probs)
    assert scores_with_probs.shape == (2, 3)
    assert scores_without_probs.shape == (2, 3)


def test_sapsscore_reproducibility() -> None:
    """Test SAPSScore reproducibility with same seed."""
    model = MockModel()

    score1 = SAPSScore(model, lambda_val=0.1, random_state=42)
    score2 = SAPSScore(model, lambda_val=0.1, random_state=42)

    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])

    scores1 = score1.calibration_nonconformity(x_cal, y_cal)
    scores2 = score2.calibration_nonconformity(x_cal, y_cal)

    assert np.allclose(scores1, scores2)

    score3 = SAPSScore(model, lambda_val=0.1, random_state=123)
    scores3 = score3.calibration_nonconformity(x_cal, y_cal)

    assert not np.allclose(scores1, scores3)


def test_saps_score_func_edge_case_single_class() -> None:
    """Test saps_score_func with single class."""
    probs = np.array([1.0], dtype=float)
    label = 0

    score: float = saps_score_func(probs, label, lambda_val=0.1, u=0.5)

    # single score has always rank 1
    assert np.isclose(score, 0.5, atol=1e-10)


def test_saps_score_func_edge_case_ties() -> None:
    """Test saps_score_func with tied probabilities."""
    probs = np.array([0.4, 0.4, 0.2], dtype=float)
    label = 0

    score: float = saps_score_func(probs, label, lambda_val=0.1, u=0.5)

    # rank could be 1 or 2 depending on sorting implementation
    # just check if valid
    assert score >= 0
    max_prob = np.max(probs)
    assert score <= max_prob + (len(probs) - 1) * 0.1 + 1e-6


def test_saps_score_func_batch_edge_case_large_batch() -> None:
    """Test saps_score_func_batch with large batch size."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    n_classes = 5

    # generate random probabilities
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples).astype(np.float32)
    labels = rng.integers(0, n_classes, size=n_samples)

    scores = saps_score_func(probs, labels, lambda_val=0.1)

    assert scores.shape == (n_samples,)
    assert np.all(scores >= 0)
    assert scores.dtype in (np.float32, np.float64)

    # upper bound check
    max_probs = np.max(probs, axis=1)
    max_scores = max_probs + (n_classes - 1) * 0.1
    assert np.all(scores <= max_scores + 1e-6)


def test_saps_score_func_2d_input() -> None:
    """Test saps_score_func with 2D input (single sample)."""
    probs = np.array([[0.5, 0.3, 0.2]], dtype=float)
    label = 0

    score: float = saps_score_func(probs, label, lambda_val=0.1, u=0.5)

    assert isinstance(score, float)
    assert np.isclose(score, 0.25, atol=1e-10)


def test_saps_score_func_invalid_2d_input() -> None:
    """Test saps_score_func with invalid 2D input."""
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]], dtype=float)
    label = 0

    with pytest.raises(ValueError, match=".*"):
        saps_score_func(probs, label, lambda_val=0.1, u=0.5)


def test_sapsscore_prediction_ranks_correctness() -> None:
    """Test SAPSScore prediction ranks correctness."""
    probs = np.array([0.7, 0.2, 0.1], dtype=float)
    model = MockModel(probs=probs)
    score = SAPSScore(model, lambda_val=0.1, random_state=42)

    x_test = np.array([[1, 2]])
    prediction_scores = score.predict_nonconformity(x_test)

    # for this pros, ranks should be 1,2,3
    # score calculation depends on random u, but pattern should be:
    # class 0 (rank 1) should have lowest score
    # class 1 (rank 2) should have middle score
    # class 2 (rank 3) should have highest score

    assert prediction_scores[0, 0] < prediction_scores[0, 1]
    assert prediction_scores[0, 1] < prediction_scores[0, 2]


def test_sapsscore_with_different_n_classes() -> None:
    """Test SAPSScore with different number of classes."""
    # test with 2 classes
    model_2 = MockModel(n_classes=2)
    score_2 = SAPSScore(model_2, lambda_val=0.1, random_state=42)

    # test with 5 classes
    model_5 = MockModel(n_classes=5)
    score_5 = SAPSScore(model_5, lambda_val=0.1, random_state=42)

    x_cal = np.array([[1, 2]])
    y_cal = np.array([0])

    scores_2 = score_2.calibration_nonconformity(x_cal, y_cal)
    scores_5 = score_5.calibration_nonconformity(x_cal, y_cal)

    assert scores_2.shape == (1,)
    assert scores_5.shape == (1,)


def test_saps_score_func_with_extreme_lambda() -> None:
    """Test saps_score_func with extreme lambda values."""
    probs = np.array([0.5, 0.3, 0.2], dtype=float)
    label = 2

    # lambda = 0
    score_lambda_zero: float = saps_score_func(probs, label, lambda_val=0.0, u=0.5)

    # lambda very small
    score_lambda_small: float = saps_score_func(probs, label, lambda_val=1e-10, u=0.5)

    # lambda large
    score_lambda_large: float = saps_score_func(probs, label, lambda_val=1.0, u=0.5)

    assert np.isclose(score_lambda_zero, 0.5, atol=1e-10)
    assert score_lambda_large > score_lambda_small
    assert score_lambda_large >= 0


def test_sapsscore_output_ranges() -> None:
    """Test SAPSScore output ranges."""
    model = MockModel(probs=np.array([[0.8, 0.15, 0.05]]))
    score = SAPSScore(model, lambda_val=0.1, random_state=42)

    x_test = np.array([[1, 2]])
    scores = score.predict_nonconformity(x_test)

    assert np.all(scores >= 0)

    # upper bound check
    max_prob = 0.8
    n_classes = 3
    upper_bound = max_prob + (n_classes - 1) * 0.1
    assert np.all(scores <= upper_bound + 1e-6)
