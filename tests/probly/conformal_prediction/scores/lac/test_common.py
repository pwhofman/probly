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
        self.probs: npt.NDArray[np.floating] = probs if probs is not None else np.array([[0.33, 0.33, 0.33]])

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

    all_scores: npt.NDArray[np.floating] = lac_score_func(probs)

    assert all_scores.shape == (2, 3)
    # LAC scores are 1 - probability
    assert np.allclose(all_scores, 1 - probs)
    assert np.all(all_scores >= 0)
    assert np.all(all_scores <= 1)


def test_accretive_completion() -> None:
    """Test accretive completion functionality."""
    # Test with empty sets
    prediction_sets = np.array(
        [
            [False, False, False],  # empty - should be completed
            [True, False, False],  # already has one - unchanged
            [False, True, True],  # already has two - unchanged
        ],
    )

    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],  # class 1 has highest prob (0.8)
            [0.9, 0.05, 0.05],  # class 0 has highest prob (0.9)
            [0.3, 0.4, 0.3],  # class 1 has highest prob (0.4)
        ],
    )

    completed = accretive_completion(prediction_sets, probabilities)

    # first row should have class 1 added (index 1)
    assert completed[0, 1]
    assert not completed[0, 0]
    assert not completed[0, 2]

    # second row unchanged
    assert np.array_equal(completed[1], [True, False, False])

    # third row unchanged
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

    # should be unchanged
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
    # with uniform probs 0.33, expected score is 1 - 0.33 = 0.67
    expected_score = 1 - 0.33
    assert np.allclose(calibration_scores, expected_score, atol=0.01)


def test_lacscore_prediction() -> None:
    """Test LACScore prediction."""
    model = MockModel()
    score = LACScore(model)

    x_test = np.array([[1, 2], [3, 4]])
    prediction_scores = score.predict_nonconformity(x_test)

    assert prediction_scores.shape == (2, 3)  # (n_samples, n_classes)
    # should be 1 - probabilities
    assert np.allclose(prediction_scores, 1 - 0.33, atol=0.01)


def test_lac_score_func_single_sample_2d() -> None:
    """Test lac_score_func with single sample as 2D array."""
    probs = np.array([[0.5, 0.3, 0.2]])  # shape (1, 3)
    scores = lac_score_func(probs)
    assert scores.shape == (1, 3)


def test_lac_score_func_edge_case_single_sample() -> None:
    """Test lac_score_func with single sample."""
    probs = np.array([[0.5, 0.3, 0.2]])
    all_scores: npt.NDArray[np.floating] = lac_score_func(probs)

    assert all_scores.shape == (1, 3), f"expected shape (1, 3), got {all_scores.shape}"
    assert np.allclose(all_scores, 1 - probs)
    assert np.all(all_scores >= 0)
    assert np.all(all_scores <= 1)


def test_lac_score_func_edge_case_large_batch() -> None:
    """Test lac_score_func with large batch."""
    rng = np.random.default_rng(42)
    probs = rng.dirichlet([1, 1, 1], size=1000).astype(np.float32)
    all_scores: npt.NDArray[np.floating] = lac_score_func(probs)

    assert all_scores.shape == (1000, 3), f"expected shape (1000, 3), got {all_scores.shape}"
    assert np.allclose(all_scores, 1 - probs)
    assert np.all(all_scores >= 0)
    assert np.all(all_scores <= 1)


def test_lac_score_func_output_types() -> None:
    """Test lac_score_func returns correct types."""
    probs = np.array([[0.5, 0.3, 0.2]])
    all_scores: npt.NDArray[np.floating] = lac_score_func(probs)

    assert isinstance(all_scores, np.ndarray), f"expected np.ndarray, got {type(all_scores)}"
    assert all_scores.dtype in [np.float32, np.float64], f"expected float dtype, got {all_scores.dtype}"


def test_lac_score_func_boundary_conditions() -> None:
    """Test lac_score_func with boundary probability distributions."""
    # Test with uniform probabilities
    probs_uniform = np.array([[0.33, 0.33, 0.34]])
    all_scores_uniform: npt.NDArray[np.floating] = lac_score_func(probs_uniform)
    assert all_scores_uniform.shape == (1, 3)
    assert np.allclose(all_scores_uniform, 1 - probs_uniform)
    assert np.all(all_scores_uniform >= 0)
    assert np.all(all_scores_uniform <= 1)

    # Test with concentrated probabilities (one class has high prob)
    probs_concentrated = np.array([[0.9, 0.05, 0.05]])
    all_scores_concentrated: npt.NDArray[np.floating] = lac_score_func(probs_concentrated)
    assert all_scores_concentrated.shape == (1, 3)
    assert np.allclose(all_scores_concentrated, 1 - probs_concentrated)
    assert np.all(all_scores_concentrated >= 0)
    assert np.all(all_scores_concentrated <= 1)

    # Test with one class having probability 1
    probs_extreme = np.array([[1.0, 0.0, 0.0]])
    all_scores_extreme: npt.NDArray[np.floating] = lac_score_func(probs_extreme)
    assert all_scores_extreme.shape == (1, 3)
    assert np.allclose(all_scores_extreme, 1 - probs_extreme)
    assert np.all(all_scores_extreme >= 0)
    assert np.all(all_scores_extreme <= 1)


def test_lac_score_func_multiple_classes() -> None:
    """Test lac_score_func with different numbers of classes."""
    # Test with 2 classes
    probs_2 = np.array([[0.6, 0.4]])
    all_scores_2: npt.NDArray[np.floating] = lac_score_func(probs_2)
    assert all_scores_2.shape == (1, 2)
    assert np.allclose(all_scores_2, 1 - probs_2)

    # Test with 5 classes
    probs_5 = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    all_scores_5: npt.NDArray[np.floating] = lac_score_func(probs_5)
    assert all_scores_5.shape == (1, 5)
    assert np.allclose(all_scores_5, 1 - probs_5)

    # Test with 10 classes
    probs_10 = np.ones((1, 10)) / 10
    all_scores_10: npt.NDArray[np.floating] = lac_score_func(probs_10)
    assert all_scores_10.shape == (1, 10)
    assert np.allclose(all_scores_10, 1 - probs_10)


def test_lacscore_prediction_output_types() -> None:
    """Test LACScore prediction output types and shapes."""
    model = MockModel()
    score = LACScore(model)

    x_test = np.array([[1, 2], [3, 4]])
    prediction_scores = score.predict_nonconformity(x_test)

    assert isinstance(prediction_scores, np.ndarray), f"expected np.ndarray, got {type(prediction_scores)}"
    assert prediction_scores.dtype in [np.float32, np.float64], f"expected float dtype, got {prediction_scores.dtype}"
    assert prediction_scores.shape == (2, 3), f"expected shape (2, 3), got {prediction_scores.shape}"


def test_accretive_completion_all_empty() -> None:
    """Test accretive completion when all sets are empty."""
    prediction_sets = np.array(
        [
            [False, False, False],
            [False, False, False],
        ],
    )

    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.9, 0.05, 0.05],
        ],
    )

    completed = accretive_completion(prediction_sets, probabilities)

    # both should have the highest probability class added
    assert completed[0, 1]  # class 1 has highest prob (0.8)
    assert completed[1, 0]  # class 0 has highest prob (0.9)


def test_accretive_completion_single_sample() -> None:
    """Test accretive completion with single sample."""
    prediction_sets = np.array([[False, False, False]])
    probabilities = np.array([[0.2, 0.7, 0.1]])

    completed = accretive_completion(prediction_sets, probabilities)

    # should add class 1 (highest probability)
    assert completed[0, 1]
    assert not completed[0, 0]
    assert not completed[0, 2]


def test_lacscore_with_different_label_values() -> None:
    """Test LACScore with different label values."""
    model = MockModel(probs=np.array([[0.33, 0.33, 0.34], [0.25, 0.5, 0.25]]))
    score = LACScore(model)

    x_cal = np.array([[1, 2], [3, 4]])
    # different label values
    y_cal = np.array([0, 2])

    calibration_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert calibration_scores.shape == (2,)
    assert np.all(calibration_scores >= 0)
    assert np.all(calibration_scores <= 1)
