"""Tests for conformal wrappers through generic representers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from probly.calibrator import calibrate
from probly.conformal_scores import (
    APSScore,
    cqr_r_score,
    cqr_score,
    lac_score,
)
from probly.method.conformal import (
    conformal_absolute_error,
    conformal_aps,
    conformal_cqr,
    conformal_cqr_r,
    conformal_lac,
    conformal_uacqr,
)
from probly.representer import representer


class DummyClassifier(BaseEstimator):
    """Simple classifier that returns fixed class probabilities."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = np.array([0.6, 0.3, 0.1], dtype=float)
        return np.repeat(probs[None, :], len(x), axis=0)


class DummyRegressor(BaseEstimator):
    """Simple regressor returning a linear trend."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x[:, 0] + 1.0


class DummyQuantileRegressor(BaseEstimator):
    """Simple quantile regressor returning fixed-width intervals."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        center = x[:, 0]
        return np.column_stack([center - 1.0, center + 1.0])


class DummyEnsembleQuantileRegressor(BaseEstimator):
    """Simple ensemble quantile regressor for UACQR."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        center = x[:, 0]
        lower = np.stack([center - 1.2, center - 1.0, center - 0.8], axis=0)
        upper = np.stack([center + 0.8, center + 1.0, center + 1.2], axis=0)
        return np.stack([lower, upper], axis=-1)


def test_classification_prediction_set_with_function_score() -> None:
    x_calib = np.arange(6.0).reshape(3, 2)
    y_calib = np.array([0, 1, 2], dtype=int)
    x_test = np.arange(8.0).reshape(4, 2)

    predictor = conformal_lac(DummyClassifier())
    calibrated = calibrate(predictor, 0.2, y_calib, x_calib)

    output = representer(calibrated).predict(x_test)
    expected_scores = lac_score(calibrated.predict(x_test), None)
    expected = expected_scores <= calibrated.conformal_quantile

    np.testing.assert_array_equal(output.array, expected)


def test_classification_prediction_set_with_callable_class_score() -> None:
    x_calib = np.arange(6.0).reshape(3, 2)
    y_calib = np.array([0, 1, 2], dtype=int)
    x_test = np.arange(8.0).reshape(4, 2)

    predictor = conformal_aps(DummyClassifier(), randomized=False)
    score = APSScore(randomized=False)
    calibrated = calibrate(predictor, 0.2, y_calib, x_calib)

    output = representer(calibrated).predict(x_test)
    expected_scores = score(calibrated.predict(x_test), None)
    expected = expected_scores <= calibrated.conformal_quantile

    np.testing.assert_array_equal(output.array, expected)


def test_regression_prediction_set_with_absolute_error_score() -> None:
    x_calib = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_calib = np.array([1.2, 2.7, 4.8, 7.1])
    x_test = np.array([[0.5], [1.5], [2.5]])

    predictor = conformal_absolute_error(DummyRegressor())
    calibrated = calibrate(predictor, 0.2, y_calib, x_calib)

    output = representer(calibrated).predict(x_test)
    prediction = predictor.predict(x_test)
    expected_lower = prediction - calibrated.conformal_quantile
    expected_upper = prediction + calibrated.conformal_quantile

    np.testing.assert_allclose(output.array[:, 0], expected_lower)
    np.testing.assert_allclose(output.array[:, 1], expected_upper)


@pytest.mark.parametrize(
    ("wrapper", "score"),
    [
        (conformal_cqr, cqr_score),
        (conformal_cqr_r, cqr_r_score),
    ],
)
def test_quantile_prediction_set_uses_score_specific_interval_formula(
    wrapper: Callable[[BaseEstimator], object],
    score: Callable[[object, object | None], object],
) -> None:
    x_calib = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_calib = np.array([0.5, 1.3, 1.9, 2.6])
    x_test = np.array([[0.5], [1.5], [2.5]])

    base_predictor = DummyQuantileRegressor()
    predictor = wrapper(base_predictor)
    calibrated = calibrate(predictor, 0.2, y_calib, x_calib)

    output = representer(calibrated).predict(x_test)
    prediction = base_predictor.predict(x_test)
    if score is cqr_score:
        expected_lower = prediction[:, 0] - calibrated.conformal_quantile
        expected_upper = prediction[:, 1] + calibrated.conformal_quantile
    else:
        width = prediction[:, 1] - prediction[:, 0]
        expected_lower = prediction[:, 0] - calibrated.conformal_quantile * width
        expected_upper = prediction[:, 1] + calibrated.conformal_quantile * width

    np.testing.assert_allclose(output.array[:, 0], expected_lower)
    np.testing.assert_allclose(output.array[:, 1], expected_upper)


def test_uacqr_prediction_set_uses_ensemble_uncertainty_scaling() -> None:
    x_calib = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_calib = np.array([0.7, 1.2, 2.4, 3.1])
    x_test = np.array([[0.5], [1.5], [2.5]])

    predictor = conformal_uacqr(DummyEnsembleQuantileRegressor())
    calibrated = calibrate(predictor, 0.2, y_calib, x_calib)

    output = representer(calibrated).predict(x_test)
    prediction = predictor.predict(x_test)
    mean_prediction = prediction.mean(axis=0)
    std_prediction = prediction.std(axis=0, ddof=1)
    expected_lower = mean_prediction[:, 0] - calibrated.conformal_quantile * std_prediction[:, 0]
    expected_upper = mean_prediction[:, 1] + calibrated.conformal_quantile * std_prediction[:, 1]

    np.testing.assert_allclose(output.array[:, 0], expected_lower)
    np.testing.assert_allclose(output.array[:, 1], expected_upper)


def test_uncalibrated_conformal_predictor_raises() -> None:
    x_test = np.array([[0.25], [0.75]])

    predictor = conformal_absolute_error(DummyRegressor())

    with pytest.raises(ValueError, match="not calibrated"):
        _ = representer(predictor).predict(x_test)
