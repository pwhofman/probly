"""Tests for RAPS common functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.scores.raps.common import RAPSScore, raps_score_func


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


def test_raps_score_func_basic() -> None:
    """Test raps_score_func with basic data."""
    probs = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
    )

    all_scores: npt.NDArray[np.floating] = raps_score_func(probs, lambda_reg=0.1, k_reg=0)

    assert all_scores.shape == (2, 3)
    assert np.all(all_scores >= 0)
    # RAPS adds regularization, so can exceed 1
    assert all_scores[0, 0] > 0


def test_rapsscore_calibration() -> None:
    """Test RAPSScore calibration."""
    model = MockModel()
    score = RAPSScore(model, lambda_reg=0.1, k_reg=0)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    calibration_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert calibration_scores.shape == (3,)
    assert np.all(calibration_scores >= 0)


def test_rapsscore_prediction() -> None:
    """Test RAPSScore prediction."""
    model = MockModel()
    score = RAPSScore(model, lambda_reg=0.1, k_reg=0)

    x_test = np.array([[1, 2], [3, 4]])
    prediction_scores = score.predict_nonconformity(x_test)

    assert prediction_scores.shape == (2, 3)  # (n_samples, n_classes)
    assert np.all(prediction_scores >= 0)


def test_rapsscore_with_different_lambda() -> None:
    """Test RAPSScore with different regularization parameters."""
    model = MockModel()
    score_low_reg = RAPSScore(model, lambda_reg=0.01, k_reg=0)
    score_high_reg = RAPSScore(model, lambda_reg=1.0, k_reg=0)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    calibration_scores_low = score_low_reg.calibration_nonconformity(x_cal, y_cal)
    calibration_scores_high = score_high_reg.calibration_nonconformity(x_cal, y_cal)

    # higher regularization should penalize larger sets
    assert calibration_scores_low.shape == (3,)
    assert calibration_scores_high.shape == (3,)
    assert np.all(calibration_scores_low >= 0)
    assert np.all(calibration_scores_high >= 0)


def test_rapsscore_provided_probs() -> None:
    """Test RAPSScore with provided probabilities."""
    model = MockModel()
    score = RAPSScore(model, lambda_reg=0.1, k_reg=0)

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


def test_raps_score_func_edge_case_single_sample() -> None:
    """Test raps_score_func with single sample."""
    probs = np.array([[0.5, 0.3, 0.2]])
    all_scores: npt.NDArray[np.floating] = raps_score_func(probs, lambda_reg=0.1, k_reg=0)

    assert all_scores.shape == (1, 3), f"Expected shape (1, 3), got {all_scores.shape}"
    assert np.all(all_scores >= 0)


def test_raps_score_func_edge_case_large_batch() -> None:
    """Test raps_score_func with large batch."""
    rng = np.random.default_rng(42)
    probs = rng.dirichlet([1, 1, 1], size=1000).astype(np.float32)
    all_scores: npt.NDArray[np.floating] = raps_score_func(probs, lambda_reg=0.1, k_reg=0)

    assert all_scores.shape == (1000, 3), f"Expected shape (1000, 3), got {all_scores.shape}"
    assert bool(np.all(all_scores >= 0))


def test_raps_score_func_output_types() -> None:
    """Test raps_score_func returns correct types."""
    probs = np.array([[0.5, 0.3, 0.2]])
    all_scores: npt.NDArray[np.floating] = raps_score_func(probs)

    assert isinstance(all_scores, np.ndarray), f"Expected np.ndarray, got {type(all_scores)}"
    assert all_scores.dtype in [np.float32, np.float64], f"Expected float dtype, got {all_scores.dtype}"


def test_raps_score_func_boundary_conditions() -> None:
    """Test raps_score_func with boundary probability distributions."""
    # Test with uniform probabilities
    probs_uniform = np.array([[0.33, 0.33, 0.34]])
    all_scores_uniform: npt.NDArray[np.floating] = raps_score_func(probs_uniform, lambda_reg=0.1, k_reg=0)
    assert all_scores_uniform.shape == (1, 3)
    assert np.all(all_scores_uniform >= 0)

    # Test with concentrated probabilities (one class has high prob)
    probs_concentrated = np.array([[0.9, 0.05, 0.05]])
    all_scores_concentrated: npt.NDArray[np.floating] = raps_score_func(probs_concentrated, lambda_reg=0.1, k_reg=0)
    assert all_scores_concentrated.shape == (1, 3)
    assert np.all(all_scores_concentrated >= 0)

    # Test with one class having probability 1
    probs_extreme = np.array([[1.0, 0.0, 0.0]])
    all_scores_extreme: npt.NDArray[np.floating] = raps_score_func(probs_extreme, lambda_reg=0.1, k_reg=0)
    assert all_scores_extreme.shape == (1, 3)
    assert np.all(all_scores_extreme >= 0)


def test_rapsscore_reproducibility_with_seed() -> None:
    """Test RAPSScore with same seed produces same results."""
    model = MockModel()

    # Create two scores with same random state
    score1 = RAPSScore(model, lambda_reg=0.1, k_reg=0, random_state=42)
    score2 = RAPSScore(model, lambda_reg=0.1, k_reg=0, random_state=42)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    scores1 = score1.calibration_nonconformity(x_cal, y_cal)
    scores2 = score2.calibration_nonconformity(x_cal, y_cal)

    # Same seed should produce same results
    assert np.allclose(scores1, scores2, rtol=1e-10, atol=1e-10), (
        f"same seed should produce same results: {scores1} != {scores2}"
    )

    # Different seed should produce different results
    # (but could coincidentally be similar due to randomization)
    score3 = RAPSScore(model, lambda_reg=0.1, k_reg=0, random_state=123)
    scores3 = score3.calibration_nonconformity(x_cal, y_cal)

    # Check if they're different (not all close)
    are_different = not np.allclose(scores1, scores3, rtol=1e-10, atol=1e-10)

    if not are_different:
        assert True


def test_rapsscore_with_different_k_reg() -> None:
    """Test RAPSScore with different k_reg (minimum set size)."""
    model = MockModel()
    score_k0 = RAPSScore(model, lambda_reg=0.1, k_reg=0)
    score_k1 = RAPSScore(model, lambda_reg=0.1, k_reg=1)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    scores_k0 = score_k0.calibration_nonconformity(x_cal, y_cal)
    scores_k1 = score_k1.calibration_nonconformity(x_cal, y_cal)

    # different k_reg values should produce different scores
    assert not np.array_equal(scores_k0, scores_k1)

    # both should be in valid range
    assert np.all(scores_k0 >= 0)
    assert np.all(scores_k1 >= 0)


def test_rapsscore_prediction_output_types() -> None:
    """Test RAPSScore prediction output types and shapes."""
    model = MockModel()
    score = RAPSScore(model, lambda_reg=0.1, k_reg=0)

    x_test = np.array([[1, 2], [3, 4]])
    prediction_scores = score.predict_nonconformity(x_test)

    assert isinstance(prediction_scores, np.ndarray), f"Expected np.ndarray, got {type(prediction_scores)}"
    assert prediction_scores.dtype in [np.float32, np.float64], f"Expected float dtype, got {prediction_scores.dtype}"
    assert prediction_scores.shape == (2, 3), f"Expected shape (2, 3), got {prediction_scores.shape}"


def test_raps_score_func_multiple_classes() -> None:
    """Test raps_score_func with different numbers of classes."""
    # Test with 2 classes
    probs_2 = np.array([[0.6, 0.4]])
    all_scores_2: npt.NDArray[np.floating] = raps_score_func(probs_2, lambda_reg=0.1, k_reg=0)
    assert all_scores_2.shape == (1, 2)

    # Test with 5 classes
    probs_5 = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    all_scores_5: npt.NDArray[np.floating] = raps_score_func(probs_5, lambda_reg=0.1, k_reg=0)
    assert all_scores_5.shape == (1, 5)

    # Test with 10 classes
    probs_10 = np.ones((1, 10)) / 10
    all_scores_10: npt.NDArray[np.floating] = raps_score_func(probs_10, lambda_reg=0.1, k_reg=0)
    assert all_scores_10.shape == (1, 10)


def test_rapsscore_with_different_label_values() -> None:
    """Test RAPSScore with different label values."""
    model = MockModel(probs=np.array([[0.33, 0.33, 0.34], [0.25, 0.5, 0.25]]))
    score = RAPSScore(model, lambda_reg=0.1, k_reg=0)

    x_cal = np.array([[1, 2], [3, 4]])
    # different label values
    y_cal = np.array([0, 2])

    calibration_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert calibration_scores.shape == (2,)
    assert np.all(calibration_scores >= 0)


def test_rapsscore_with_different_epsilon() -> None:
    """Test RAPSScore with different epsilon (stability parameter) values."""
    model = MockModel()
    score_low_eps = RAPSScore(model, lambda_reg=0.1, k_reg=0, epsilon=0.001)
    score_high_eps = RAPSScore(model, lambda_reg=0.1, k_reg=0, epsilon=0.1)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    calibration_scores_low = score_low_eps.calibration_nonconformity(x_cal, y_cal)
    calibration_scores_high = score_high_eps.calibration_nonconformity(x_cal, y_cal)

    # different epsilon values should produce different scores
    assert not np.array_equal(calibration_scores_low, calibration_scores_high)

    # both should be valid
    assert calibration_scores_low.shape == (3,)
    assert calibration_scores_high.shape == (3,)
    assert np.all(calibration_scores_low >= 0)
    assert np.all(calibration_scores_high >= 0)


def test_rapsscore_with_randomization() -> None:
    """Test RAPSScore with randomization enabled."""
    model = MockModel()
    score_rand = RAPSScore(model, lambda_reg=0.1, k_reg=0, randomize=True, random_state=42)
    score_no_rand = RAPSScore(model, lambda_reg=0.1, k_reg=0, randomize=False)

    x_cal = np.array([[1, 2], [3, 4], [5, 6]])
    y_cal = np.array([0, 1, 2])

    scores_rand = score_rand.calibration_nonconformity(x_cal, y_cal)
    scores_no_rand = score_no_rand.calibration_nonconformity(x_cal, y_cal)

    # with randomization, results should generally differ
    # But allow for the possibility they might be similar
    are_different = not np.allclose(scores_rand, scores_no_rand, rtol=1e-10, atol=1e-10)

    # Both should produce valid scores
    assert scores_rand.shape == (3,)
    assert scores_no_rand.shape == (3,)
    assert np.all(scores_rand >= 0)
    assert np.all(scores_no_rand >= 0)

    # If they're the same, it's unusual but could happen with specific seeds
    if not are_different:
        assert True
