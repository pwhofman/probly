"""Tests for CQR common functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from probly.conformal_prediction.scores.cqr.common import CQRScore, cqr_score_func


class MockQuantileModel:
    """Mock model for testing CQR without real training."""

    def __init__(self, intervals: npt.NDArray[np.float64] | None = None) -> None:
        """Initialize mock model with specific interval outputs.

        Default interval is [4.0, 6.0].
        """
        self.intervals = intervals if intervals is not None else np.array([[4.0, 6.0]])

    def predict(self, x: object) -> npt.NDArray[np.float64]:
        """Return fixed intervals repeated for input size."""
        n = len(x) if hasattr(x, "__len__") else 1
        # If we have a single interval pair, repeat it for all samples
        if self.intervals.shape[0] == 1 and n > 1:
            return np.repeat(self.intervals, n, axis=0)
        return self.intervals

    def __call__(self, x: object) -> npt.NDArray[np.float64]:
        """Alias for predict to mimic callable models."""
        return self.predict(x)


def test_cqr_score_func_basic() -> None:
    """Test cqr_score_func with manual calculation."""
    # Data points: inside interval, below lower bound, above upper bound
    y_true: npt.NDArray[np.floating] = np.array([5.0, 2.0, 8.0])
    # y_pred: All have interval [4.0, 6.0]
    y_pred: npt.NDArray[np.floating] = np.array([[4.0, 6.0], [4.0, 6.0], [4.0, 6.0]])
    scores: npt.NDArray[np.floating] = cqr_score_func(y_true, y_pred)

    # Expected logic (CQR formula: max(q_lo - y, y - q_hi)):
    # 1. Inside [4, 6]: max(4-5=-1, 5-6=-1) = -1
    # 2. Below: max(4-2=2, 2-6=-4) = 2
    # 3. Above: max(4-8=-4, 8-6=2) = 2
    expected: npt.NDArray[np.floating] = np.array([-1.0, 2.0, 2.0])

    assert scores.shape == (3,)
    # Negative scores are allowed in CQR, remove: assert np.all(scores >= 0)
    assert np.allclose(scores, expected, atol=1e-10)


def test_cqr_score_func_edge_case_single_sample() -> None:
    """Test cqr_score_func with a single sample (N=1)."""
    y_true: npt.NDArray[np.floating] = np.array([10.0])
    y_pred: npt.NDArray[np.floating] = np.array([[2.0, 8.0]])  # y is above upper bound
    score: npt.NDArray[np.floating] = cqr_score_func(y_true, y_pred)

    assert score.shape == (1,)
    # max(2-10=-8, 10-8=2) = 2.0
    assert np.isclose(score[0], 2.0)


def test_cqr_score_func_edge_case_large_batch() -> None:
    """Test cqr_score_func with a large batch (Stress Test)."""
    # 1000 samples, all y=0
    y_true: npt.NDArray[np.floating] = np.zeros(1000)
    # All intervals [-1, 1] -> y=0 is always inside
    y_pred: npt.NDArray[np.floating] = np.column_stack((-np.ones(1000), np.ones(1000)))
    scores: npt.NDArray[np.floating] = cqr_score_func(y_true, y_pred)

    assert scores.shape == (1000,)
    # All scores should be -1.0 since y is inside the interval [-1, 1]
    # max(-1-0=-1, 0-1=-1) = -1.0
    assert np.all(scores == -1.0)


def test_cqr_score_func_output_types() -> None:
    """Test strict output types (Float64)."""
    y_true: npt.NDArray[np.floating] = np.array([5.0])
    y_pred: npt.NDArray[np.floating] = np.array([[4.0, 6.0]])
    scores: npt.NDArray[np.floating] = cqr_score_func(y_true, y_pred)

    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float64


def test_cqrscore_calibration_flow() -> None:
    """Test the full calibration flow using MockModel."""
    model = MockQuantileModel()  # Returns [4.0, 6.0] by default
    score = CQRScore(model)
    x_cal: npt.NDArray[np.floating] = np.array([[1], [2], [3]])
    y_cal: npt.NDArray[np.floating] = np.array([5.0, 3.0, 7.0])  # Inside, Below, Above

    cal_scores: npt.NDArray[np.floating] = score.calibration_nonconformity(x_cal, y_cal)

    # handle shape if needed
    if cal_scores.ndim == 2 and cal_scores.shape[1] == 1:
        cal_scores = cal_scores.flatten()

    # Expected using CQR formula:
    # 5.0 inside [4,6]: max(4-5=-1, 5-6=-1) = -1
    # 3.0 below [4,6]: max(4-3=1, 3-6=-3) = 1
    # 7.0 above [4,6]: max(4-7=-3, 7-6=1) = 1
    expected: npt.NDArray[np.floating] = np.array([-1.0, 1.0, 1.0])

    assert cal_scores.shape == (3,)
    assert np.allclose(cal_scores, expected, atol=1e-10)


def test_cqrscore_prediction_flow() -> None:
    """Test prediction flow (returning interval widths)."""
    # Model returns intervals [4, 6] -> Width is always 2
    model = MockQuantileModel(intervals=np.array([[4.0, 6.0]]))
    score = CQRScore(model)
    x_test: npt.NDArray[np.floating] = np.array([[1], [2]])

    widths: npt.NDArray[np.floating] = score.predict_nonconformity(x_test)

    # If it returns predictions (N, 2), compute widths
    if widths.shape[1] == 2:
        widths = (widths[:, 1] - widths[:, 0]).reshape(-1, 1)

    assert widths.shape == (2, 1)  # (N, 1) required for Score protocol
    assert np.all(widths == 2.0)


def test_cqrscore_provided_predictions() -> None:
    """Test bypassing the model by providing y_pred explicitly."""
    model = MockQuantileModel()  # Would return [4, 6]
    score = CQRScore(model)
    x: npt.NDArray[np.floating] = np.array([[1]])
    y: npt.NDArray[np.floating] = np.array([10.0])

    # Force a different interval: [9, 11]
    # y=10 is inside [9, 11] -> Score must be -1 (instead of 4 via MockModel)
    forced_pred: npt.NDArray[np.floating] = np.array([[9.0, 11.0]])

    cal_score: npt.NDArray[np.floating] = score.calibration_nonconformity(x, y, y_pred=forced_pred)

    # handle shape if needed
    if cal_score.ndim == 2 and cal_score.shape[1] == 1:
        cal_score = cal_score.flatten()

    # max(9-10=-1, 10-11=-1) = -1.0
    assert cal_score[0] == -1.0


def test_cqrscore_input_validation() -> None:
    """Test that wrong shapes raise errors."""
    model = MockQuantileModel()
    score = CQRScore(model)
    x: npt.NDArray[np.floating] = np.array([1])
    y: npt.NDArray[np.floating] = np.array([1])

    # y_pred has 1 column instead of 2 -> Should raise ValueError
    bad_pred: npt.NDArray[np.floating] = np.array([[5.0]])

    with pytest.raises(ValueError, match="shape"):
        score.calibration_nonconformity(x, y, y_pred=bad_pred)
