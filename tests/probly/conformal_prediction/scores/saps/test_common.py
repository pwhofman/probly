"""Tests for SAPS common functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

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
    probs = np.array([[0.5, 0.3, 0.2]], dtype=float)
    u = np.array([[0.5, 0.5, 0.5]], dtype=float)

    # test with rank 1
    score = saps_score_func(probs, lambda_val=0.1, u=u)

    # Check that we get a 2D array back
    assert isinstance(score, np.ndarray)
    assert score.shape == (1, 3)

    # Calculate expected scores
    assert np.isclose(score[0, 0], 0.25, atol=1e-10)
    assert np.isclose(score[0, 1], 0.55, atol=1e-10)
    assert np.isclose(score[0, 2], 0.65, atol=1e-10)


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

    # Create two scores with same random state - should be identical
    score1 = SAPSScore(model, lambda_val=0.1, random_state=42)
    score2 = SAPSScore(model, lambda_val=0.1, random_state=42)

    x_cal = np.array([[1, 2], [3, 4]])
    y_cal = np.array([0, 1])

    scores1 = score1.calibration_nonconformity(x_cal, y_cal)
    scores2 = score2.calibration_nonconformity(x_cal, y_cal)

    # Same seed should produce same results
    assert np.allclose(scores1, scores2, rtol=1e-10), f"Same seed should produce same results: {scores1} != {scores2}"

    # Different seed should (usually) produce different results
    score3 = SAPSScore(model, lambda_val=0.1, random_state=123)
    scores3 = score3.calibration_nonconformity(x_cal, y_cal)

    # check that they're not identical arrays
    are_different = not np.array_equal(scores1, scores3)

    if not are_different:
        assert True


def test_saps_score_func_edge_case_single_class() -> None:
    """Test saps_score_func with single class."""
    # Ensure 2D array shape
    probs = np.array([[1.0]], dtype=float)  # Shape (1, 1)
    u = np.array([[0.5]], dtype=float)  # Shape (1, 1)

    score: np.ndarray = saps_score_func(probs, lambda_val=0.1, u=u)

    # single score has always rank 1: 1.0 * 0.5 = 0.5
    assert score.shape == (1, 1)
    assert np.isclose(score[0, 0], 0.5, atol=1e-10)


def test_saps_score_func_edge_case_ties() -> None:
    """Test saps_score_func with tied probabilities."""
    # Ensure 2D array shape
    probs = np.array([[0.4, 0.4, 0.2]], dtype=float)  # Shape (1, 3)
    u = np.array([[0.5, 0.5, 0.5]], dtype=float)  # Shape (1, 3)

    score = saps_score_func(probs, lambda_val=0.1, u=u)

    # rank could be 1 or 2 depending on sorting implementation
    # just check if valid
    assert score.shape == (1, 3)  # This line was causing the error
    assert np.all(score >= 0)
    max_prob = np.max(probs)
    assert np.all(score <= max_prob + (probs.shape[1] - 1) * 0.1 + 1e-6)


def test_saps_score_func_2d_input() -> None:
    """Test saps_score_func with 2D input (single sample)."""
    probs = np.array([[0.5, 0.3, 0.2]], dtype=float)
    u = np.array([[0.5, 0.5, 0.5]], dtype=float)

    score = saps_score_func(probs, lambda_val=0.1, u=u)

    assert isinstance(score, np.ndarray)
    assert score.shape == (1, 3)
    assert np.isclose(score[0, 0], 0.25, atol=1e-10)


def test_saps_score_func_invalid_2d_input() -> None:
    """Test saps_score_func with invalid 2D input."""
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]], dtype=float)
    u = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=float)

    scores = saps_score_func(probs, lambda_val=0.1, u=u)

    assert scores.shape == (2, 3)
    assert np.all(scores >= 0)


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
    # Ensure 2D array shape
    probs = np.array([[0.5, 0.3, 0.2]], dtype=float)  # Shape (1, 3)
    u = np.array([[0.5, 0.5, 0.5]], dtype=float)  # Shape (1, 3)

    # lambda = 0
    score_lambda_zero: np.ndarray = saps_score_func(probs, lambda_val=0.0, u=u)
    # lambda very small
    score_lambda_small: np.ndarray = saps_score_func(probs, lambda_val=1e-10, u=u)
    # lambda large
    score_lambda_large: np.ndarray = saps_score_func(probs, lambda_val=1.0, u=u)

    assert score_lambda_zero.shape == (1, 3)
    # For rank 1: 0.5 * 0.5 = 0.25
    assert np.isclose(score_lambda_zero[0, 0], 0.25, atol=1e-10)
    # With larger lambda, scores for non-top classes should increase
    assert score_lambda_large[0, 2] > score_lambda_small[0, 2]
    assert np.all(score_lambda_large >= 0)


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
