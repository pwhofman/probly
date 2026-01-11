"""Tests for the AbsoluteErrorScore."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from probly.conformal_prediction.methods.split import SplitConformalRegressor
from probly.conformal_prediction.scores.absolute_error.common import AbsoluteErrorScore


class ConstantModel:
    """A dummy model that always predicts zero."""

    # given input x, always return 0
    def __call__(self, x: Sequence[Any]) -> np.ndarray:
        """Return array of zeros with same length as input."""
        return np.zeros(len(x))


def test_regression_works() -> None:
    """Test that SplitConformalRegressor works with AbsoluteErrorScore and the new construct_intervals protocol."""
    # generate synthetic calibration and test data
    n_cal = 100
    n_test = 10

    rng = np.random.default_rng(42)

    # random X values
    x_cal = rng.random((n_cal, 5))
    x_test = rng.random((n_test, 5))

    # random y values for calibration
    y_cal = rng.random(n_cal)

    # setup model, score, and regressor
    model = ConstantModel()
    score = AbsoluteErrorScore(model)  # Using the new baseline score
    regressor = SplitConformalRegressor(model, score)  # Using the refactored regressor

    # calibrate
    alpha = 0.1
    regressor.calibrate(x_cal, y_cal, alpha)

    # check that threshold is set
    assert regressor.is_calibrated
    assert regressor.threshold is not None
    assert regressor.threshold > 0

    # predict intervals
    intervals = regressor.predict(x_test, alpha)

    # shape of intervals
    assert intervals.shape == (n_test, 2)

    # check that lower bounds are less than upper bounds
    assert np.all(intervals[:, 0] <= intervals[:, 1])

    # with AbsoluteErrorScore and model prediction 0, interval must be [-q, +q].
    q = regressor.threshold

    # check if left column is approximately -q
    np.testing.assert_allclose(intervals[:, 0], -q)

    # check if right column is approximately +q
    np.testing.assert_allclose(intervals[:, 1], q)


def test_absolute_error_construct_intervals_symmetric() -> None:
    """Test that construct_intervals creates symmetric intervals [y-q, y+q]."""
    model = ConstantModel()
    score = AbsoluteErrorScore(model)

    y_hat = np.array([1.0, 2.0, 3.0, 4.0])
    threshold = 0.5

    intervals = score.construct_intervals(y_hat, threshold)

    # check symmetry: [y - threshold, y + threshold]
    expected_lower = y_hat - threshold
    expected_upper = y_hat + threshold

    np.testing.assert_allclose(intervals[:, 0], expected_lower)
    np.testing.assert_allclose(intervals[:, 1], expected_upper)

    # verify lower < upper
    assert np.all(intervals[:, 0] < intervals[:, 1])


def test_forward_shapes() -> None:
    """Test that the shapes of calibration nonconformity scores are correct."""
    model = ConstantModel()
    score = AbsoluteErrorScore(model)

    # calibration data
    x_cal = [[1, 2], [3, 4], [5, 6]]
    y_cal = [1.0, 2.0, 3.0]

    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    # check shape
    assert cal_scores.shape == (3,)

    # check values
    expected_scores = np.array([1.0, 2.0, 3.0])  # |y - 0|
    np.testing.assert_allclose(cal_scores, expected_scores)


def test_output_types() -> None:
    """Test that the output types of calibration nonconformity scores are correct."""
    model = ConstantModel()
    score = AbsoluteErrorScore(model)

    # calibration data
    x_cal = [[1, 2], [3, 4]]
    y_cal = [0.5, 1.5]

    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    # check type
    assert isinstance(cal_scores, np.ndarray)
    assert cal_scores.dtype == np.float64


def test_absolute_error_edge_case_single_sample() -> None:
    """Test AbsoluteErrorScore with single sample."""
    model = ConstantModel()
    score = AbsoluteErrorScore(model)

    # calibration with single sample
    x_cal = [[1, 2]]
    y_cal = [1.5]
    cal_scores = score.calibration_nonconformity(x_cal, y_cal)

    assert cal_scores.shape == (1,)
    np.testing.assert_allclose(cal_scores, [1.5])  # |1.5 - 0|

    # construct intervals with single sample
    y_hat = np.array([2.0])
    intervals = score.construct_intervals(y_hat, threshold=1.0)

    assert intervals.shape == (1, 2)
    np.testing.assert_allclose(intervals[0], [1.0, 3.0])
