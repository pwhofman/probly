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
