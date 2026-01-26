"""Tests for CQR common functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from probly.conformal_prediction.scores.cqr.common import CQRScore, cqr_score_func


# --- 1. Helper: Mock Model (Fast & Independent) ---
class MockQuantileModel:
    """Mock model for testing CQR without real training."""

    def __init__(self, intervals: npt.NDArray[np.float64] | None = None) -> None:
        """Initialize mock model with specific interval outputs.

        Default interval is [4.0, 6.0].
        """
        self.intervals = intervals if intervals is not None else np.array([[4.0, 6.0]])

    # FIX: 'Any' durch 'object' ersetzt, um ANN401 zu lösen
    def predict(self, x: object) -> npt.NDArray[np.float64]:
        """Return fixed intervals repeated for input size."""
        n = len(x) if hasattr(x, "__len__") else 1
        # If we have a single interval pair, repeat it for all samples
        if self.intervals.shape[0] == 1 and n > 1:
            return np.repeat(self.intervals, n, axis=0)
        return self.intervals

    # FIX: 'Any' durch 'object' ersetzt
    def __call__(self, x: object) -> npt.NDArray[np.float64]:
        """Alias for predict to mimic callable models."""
        return self.predict(x)


# --- 2. Tests for Core Mathematical Logic ---


def test_cqr_score_func_basic() -> None:
    """Test cqr_score_func with manual calculation."""
    # FIX: Kommentar umformuliert, um ERA001 zu beheben
    # Data points: inside interval, below lower bound, above upper bound
    y_true = np.array([5.0, 2.0, 8.0])

    # y_pred: All have interval [4.0, 6.0]
    y_pred = np.array([[4.0, 6.0], [4.0, 6.0], [4.0, 6.0]])

    scores = cqr_score_func(y_true, y_pred)

    # Expected logic:
    # 1. Inside [4, 6] -> Score 0
    # 2. Below (4 - 2) -> Score 2
    # 3. Above (8 - 6) -> Score 2
    expected = np.array([0.0, 2.0, 2.0])

    assert scores.shape == (3,)
    assert np.all(scores >= 0)
    assert np.allclose(scores, expected, atol=1e-10)


def test_cqr_score_func_edge_case_single_sample() -> None:
    """Test cqr_score_func with a single sample (N=1)."""
    y_true = np.array([10.0])
    y_pred = np.array([[2.0, 8.0]])  # y is above upper bound

    score = cqr_score_func(y_true, y_pred)

    assert score.shape == (1,)
    # 10.0 - 8.0 = 2.0
    assert np.isclose(score[0], 2.0)


def test_cqr_score_func_edge_case_large_batch() -> None:
    """Test cqr_score_func with a large batch (Stress Test)."""
    # 1000 samples, all y=0
    y_true = np.zeros(1000)
    # All intervals [-1, 1] -> y=0 is always inside
    y_pred = np.column_stack((-np.ones(1000), np.ones(1000)))

    scores = cqr_score_func(y_true, y_pred)

    assert scores.shape == (1000,)
    # All scores should be 0.0 since y is inside the interval
    assert np.all(scores == 0.0)


def test_cqr_score_func_output_types() -> None:
    """Test strict output types (Float64)."""
    y_true = np.array([5.0])
    y_pred = np.array([[4.0, 6.0]])

    scores = cqr_score_func(y_true, y_pred)

    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float64


# --- 3. Tests for CQRScore Class (Integration) ---


def test_cqrscore_calibration_flow() -> None:
    """Test the full calibration flow using MockModel."""
    model = MockQuantileModel()  # Returns [4.0, 6.0] by default
    score = CQRScore(model)

    x_cal = np.array([[1], [2], [3]])
    y_cal = np.array([5.0, 3.0, 7.0])  # Inside, Below, Above

    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    # Expected: 0, 4-3=1, 7-6=1
    expected = np.array([0.0, 1.0, 1.0])
    assert np.allclose(cal_scores, expected)


def test_cqrscore_prediction_flow() -> None:
    """Test prediction flow (returning interval widths)."""
    # Model returns intervals [4, 6] -> Width is always 2
    model = MockQuantileModel(intervals=np.array([[4.0, 6.0]]))
    score = CQRScore(model)

    x_test = np.array([[1], [2]])
    widths = score.predict_nonconformity(x_test)

    assert widths.shape == (2, 1)  # (N, 1) required for Score protocol
    assert np.all(widths == 2.0)


def test_cqrscore_provided_predictions() -> None:
    """Test bypassing the model by providing y_pred explicitly."""
    model = MockQuantileModel()  # Would return [4, 6]
    score = CQRScore(model)

    x = np.array([[1]])
    y = np.array([10.0])

    # Force a different interval: [9, 11]
    # y=10 is inside [9, 11] -> Score must be 0 (instead of 4 via MockModel)
    forced_pred = np.array([[9.0, 11.0]])

    cal_score = score.calibration_nonconformity(x, y, y_pred=forced_pred)

    assert cal_score[0] == 0.0


def test_cqrscore_input_validation() -> None:
    """Test that wrong shapes raise errors."""
    model = MockQuantileModel()
    score = CQRScore(model)

    x = np.array([1])
    y = np.array([1])

    # y_pred has 1 column instead of 2 -> Should raise ValueError
    bad_pred = np.array([[5.0]])

    with pytest.raises(ValueError, match="shape"):
        score.calibration_nonconformity(x, y, y_pred=bad_pred)
