"""Predictor transformation contracts for sklearn estimators."""

from __future__ import annotations

import numpy as np
from packaging.version import Version
import pytest

from probly.method.calibration import platt_scaling, temperature_scaling, vector_scaling
from probly.method.conformal import conformal_lac
from probly.method.efficient_credal_prediction import efficient_credal_prediction
from probly.method.mahalanobis import mahalanobis
from probly.predictor import predict
from probly.representation.distribution.numpy_categorical import NumpyProbabilityCategoricalDistribution

pytest.importorskip("sklearn")
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

_NO_TEMPERATURE_SCALING = pytest.mark.skipif(
    Version(sklearn.__version__) < Version("1.8.0"), reason="sklearn temperature scaling requires version 1.8.0"
)


def _fitted_classifier(num_classes: int) -> tuple[LogisticRegression, np.ndarray]:
    x, y = make_classification(n_samples=120, n_features=6, n_informative=4, n_classes=num_classes, random_state=0)
    return LogisticRegression().fit(x, y), x


def _assert_predicts_probabilities(base: LogisticRegression, x: np.ndarray) -> None:
    prediction = predict(base, x)
    assert isinstance(prediction, NumpyProbabilityCategoricalDistribution)
    np.testing.assert_allclose(prediction.probabilities, base.predict_proba(x))


@pytest.mark.parametrize(
    "transformation",
    [
        pytest.param(temperature_scaling, marks=_NO_TEMPERATURE_SCALING),
        vector_scaling,
        mahalanobis,
        efficient_credal_prediction,
    ],
)
@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("predict_first", [False, True])
def test_logit_transformation_keeps_base_prediction(transformation, num_classes: int, predict_first: bool) -> None:
    """A method that needs logits does not change what predict returns for its base estimator."""
    base, x = _fitted_classifier(num_classes)
    if predict_first:
        predict(base, x)

    transformation(base)

    _assert_predicts_probabilities(base, x)


def test_binary_logit_transformation_keeps_base_prediction() -> None:
    """Platt scaling does not change what predict returns for its base estimator."""
    base, x = _fitted_classifier(2)
    predict(base, x)

    platt_scaling(base)

    _assert_predicts_probabilities(base, x)


def test_logit_transformation_after_probabilistic_transformation() -> None:
    """An estimator already wrapped by a conformal method can still be passed to a method that needs logits."""
    base, x = _fitted_classifier(3)
    conformal_lac(base)

    vector_scaling(base)

    _assert_predicts_probabilities(base, x)
