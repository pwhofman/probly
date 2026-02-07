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
        self.probs: npt.NDArray[np.floating] = probs if probs is not None else np.array([[0.33, 0.33, 0.33]])

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

    all_scores: npt.NDArray[np.floating] = aps_score_func(probs)

    assert all_scores.shape == (2, 3)
    assert np.all(all_scores >= 0)
    assert np.all(all_scores <= 1)

    expected_first = np.array([0.5, 0.8, 1.0])
    assert np.allclose(all_scores[0], expected_first, atol=1e-10)


def test_apsscore_calibration() -> None:
    """Test APSScore calibration."""
    model = MockModel()
    score = APSScore(model, randomize=False)

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
    score = APSScore(model, randomize=False)

    x_test = np.array([[1, 2], [3, 4]])

    prediction_scores = score.predict_nonconformity(x_test)

    assert prediction_scores.shape == (2, 3)  # (n_samples, n_classes)
    assert np.all(prediction_scores >= 0)
    assert np.all(prediction_scores <= 1)


def test_apsscore_with_randomization() -> None:
    """Test APSScore with randomization enabled."""
    model = MockModel()
    score = APSScore(model, randomize=True)

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

    # Test with provided probabilities
    scores_with_probs = score.predict_nonconformity(x_test, probs=provided_probs)

    # without provided probs, uses model's probs
    scores_without_probs = score.predict_nonconformity(x_test)

    # should differ since probs are different
    assert not np.allclose(scores_with_probs, scores_without_probs)
    assert scores_with_probs.shape == (2, 3)
    assert scores_without_probs.shape == (2, 3)


def test_aps_score_func_single_sample_2d() -> None:
    """Test aps_score_func with single sample as 2D array."""
    probs = np.array([[0.5, 0.3, 0.2]])  # 2D with shape (1, 3)
    scores = aps_score_func(probs)
    assert scores.shape == (1, 3)


def test_aps_score_func_edge_case_single_sample() -> None:
    """Test aps_score_func with single sample."""
    probs = np.array([[0.5, 0.3, 0.2]])
    all_scores: npt.NDArray[np.floating] = aps_score_func(probs)

    assert all_scores.shape == (1, 3), f"Expected shape (1, 3), got {all_scores.shape}"
    assert np.all(all_scores >= 0)
    assert bool(np.all(all_scores <= 1 + 1e-6))


def test_aps_score_func_edge_case_large_batch() -> None:
    """Test aps_score_func with large batch."""
    rng = np.random.default_rng(42)
    probs = rng.dirichlet([1, 1, 1], size=1000).astype(np.float32)
    all_scores: npt.NDArray[np.floating] = aps_score_func(probs)

    assert all_scores.shape == (1000, 3), f"Expected shape (1000, 3), got {all_scores.shape}"
    assert bool(np.all(all_scores >= 0))
    # allow small tolerance for float32 precision errors in cumsum
    assert bool(np.all(all_scores <= 1.0 + 1e-6))


def test_aps_score_func_output_types() -> None:
    """Test aps_score_func returns correct types."""
    probs = np.array([[0.5, 0.3, 0.2]])
    all_scores: npt.NDArray[np.floating] = aps_score_func(probs)

    assert isinstance(all_scores, np.ndarray), f"Expected np.ndarray, got {type(all_scores)}"
    assert all_scores.dtype in [np.float32, np.float64], f"Expected float dtype, got {all_scores.dtype}"


def test_aps_score_func_boundary_conditions() -> None:
    """Test aps_score_func with boundary probability distributions."""
    # Test with uniform probabilities
    probs_uniform = np.array([[0.33, 0.33, 0.34]])
    all_scores_uniform: npt.NDArray[np.floating] = aps_score_func(probs_uniform)

    assert all_scores_uniform.shape == (1, 3)
    assert np.all(all_scores_uniform >= 0)
    assert bool(np.all(all_scores_uniform <= 1 + 1e-6))

    # Test with concentrated probabilities (one class has high prob)
    probs_concentrated = np.array([[0.9, 0.05, 0.05]])
    all_scores_concentrated: npt.NDArray[np.floating] = aps_score_func(probs_concentrated)

    assert all_scores_concentrated.shape == (1, 3)
    assert np.all(all_scores_concentrated >= 0)
    assert bool(np.all(all_scores_concentrated <= 1 + 1e-6))

    # Test with one class having probability 1
    probs_extreme = np.array([[1.0, 0.0, 0.0]])
    all_scores_extreme: npt.NDArray[np.floating] = aps_score_func(probs_extreme)

    assert all_scores_extreme.shape == (1, 3)
    assert np.all(all_scores_extreme >= 0)
    assert bool(np.all(all_scores_extreme <= 1 + 1e-6))


def test_apsscore_randomization_reproducibility() -> None:
    """Test APSScore randomization reproducibility."""
    model = MockModel()
    score1 = APSScore(model, randomize=True, random_state=1)
    score2 = APSScore(model, randomize=True, random_state=1)
    score3 = APSScore(model, randomize=True, random_state=2)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    scores1 = score1.calibration_nonconformity(x_cal, y_cal)
    scores2 = score2.calibration_nonconformity(x_cal, y_cal)
    scores3 = score3.calibration_nonconformity(x_cal, y_cal)

    # same seed should produce same results
    assert np.allclose(scores1, scores2), "same seed should produce same results"

    # different seeds might produce same or different results
    # check only that results are valid
    assert scores3.shape == (3,)
    assert np.all(scores3 >= 0)
    assert np.all(scores3 <= 1)


def test_apsscore_with_and_without_randomization_comparison() -> None:
    """Compare APSScore with and without randomization."""
    model = MockModel()
    score_no_rand = APSScore(model, randomize=False)
    score_with_rand = APSScore(model, randomize=True)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    scores_no_rand = score_no_rand.calibration_nonconformity(x_cal, y_cal)
    scores_with_rand = score_with_rand.calibration_nonconformity(x_cal, y_cal)

    # both should be in valid range (allow tolerance for float32 precision)
    assert np.all(scores_no_rand >= 0)
    assert bool(np.all(scores_no_rand <= 1 + 1e-6))
    assert np.all(scores_with_rand >= 0)
    assert bool(np.all(scores_with_rand <= 1 + 1e-6))


def test_apsscore_prediction_output_types() -> None:
    """Test APSScore prediction output types and shapes."""
    model = MockModel()
    score = APSScore(model, randomize=False)

    x_test = np.array([[1, 2], [3, 4]])
    prediction_scores = score.predict_nonconformity(x_test)

    assert isinstance(prediction_scores, np.ndarray), f"Expected np.ndarray, got {type(prediction_scores)}"
    assert prediction_scores.dtype in [np.float32, np.float64], f"Expected float dtype, got {prediction_scores.dtype}"
    assert prediction_scores.shape == (2, 3), f"Expected shape (2, 3), got {prediction_scores.shape}"


def test_aps_score_func_multiple_classes() -> None:
    """Test aps_score_func with different numbers of classes."""
    # Test with 2 classes
    probs_2 = np.array([[0.6, 0.4]])
    all_scores_2: npt.NDArray[np.floating] = aps_score_func(probs_2)

    assert all_scores_2.shape == (1, 2)

    # Test with 5 classes
    probs_5 = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    all_scores_5: npt.NDArray[np.floating] = aps_score_func(probs_5)

    assert all_scores_5.shape == (1, 5)

    # Test with 10 classes
    probs_10 = np.ones((1, 10)) / 10
    all_scores_10: npt.NDArray[np.floating] = aps_score_func(probs_10)

    assert all_scores_10.shape == (1, 10)


def test_apsscore_with_different_label_values() -> None:
    """Test APSScore with different label values."""
    model = MockModel(probs=np.array([[0.33, 0.33, 0.34], [0.25, 0.5, 0.25]]))
    score = APSScore(model, randomize=False)

    x_cal = np.array([[1, 2], [3, 4]])
    # different label values
    y_cal = np.array([0, 2])

    calibration_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert calibration_scores.shape == (2,)
    assert np.all(calibration_scores >= 0)
    assert np.all(calibration_scores <= 1)
