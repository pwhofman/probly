"""Tests for the sklearn efficient credal prediction wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

from probly.method.efficient_credal_prediction import (
    EfficientCredalPredictor,
    compute_efficient_credal_prediction_bounds,
    efficient_credal_prediction,
)
from probly.predictor import predict
from probly.representation.credal_set import ProbabilityIntervalsCredalSet
from probly.representation.distribution.numpy_categorical import NumpyProbabilityCategoricalDistribution
from probly.representer import representer

pytest.importorskip("sklearn")
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from probly.method.efficient_credal_prediction.sklearn import SklearnEfficientCredalPredictor


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(n_samples=200, n_features=6, n_informative=4, n_classes=3, random_state=0)


@pytest.fixture
def classifier(data: tuple[np.ndarray, np.ndarray]) -> LogisticRegression:
    x, y = data
    return LogisticRegression().fit(x, y)


class TestSklearnEfficientCredalPredictor:
    """The sklearn wrapper exposes the base logits and builds interval credal sets."""

    def test_wraps_estimator_with_unset_bounds(self, classifier: LogisticRegression) -> None:
        predictor = efficient_credal_prediction(classifier)

        assert isinstance(predictor, SklearnEfficientCredalPredictor)
        assert isinstance(predictor, EfficientCredalPredictor)
        assert predictor.predictor is classifier
        assert predictor.lower is None
        assert predictor.upper is None

    def test_predict_returns_base_logits(
        self, classifier: LogisticRegression, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, _ = data
        predictor = efficient_credal_prediction(classifier)

        np.testing.assert_allclose(predict(predictor, x).logits, classifier.decision_function(x))

    def test_binary_margins_become_two_logit_columns(self) -> None:
        x, y = make_classification(n_samples=100, n_features=4, random_state=0)
        classifier = LogisticRegression().fit(x, y)
        predictor = efficient_credal_prediction(classifier)

        logits = predict(predictor, x).logits

        assert logits.shape == (len(x), 2)
        np.testing.assert_allclose(softmax(logits, axis=-1), classifier.predict_proba(x))

    def test_estimator_without_decision_function(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = data
        classifier = GaussianNB().fit(x, y)
        predictor = efficient_credal_prediction(classifier)

        np.testing.assert_allclose(softmax(predict(predictor, x).logits, axis=-1), classifier.predict_proba(x))

    def test_predict_keeps_base_prediction(
        self, classifier: LogisticRegression, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, _ = data
        predict(efficient_credal_prediction(classifier), x)

        prediction = predict(classifier, x)

        assert isinstance(prediction, NumpyProbabilityCategoricalDistribution)
        np.testing.assert_allclose(prediction.probabilities, classifier.predict_proba(x))

    def test_represent_without_bounds_raises(
        self, classifier: LogisticRegression, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, _ = data
        predictor = efficient_credal_prediction(classifier)

        with pytest.raises(RuntimeError, match="uninitialized bounds"):
            representer(predictor).predict(x)

    def test_represent_encloses_base_probabilities(
        self, classifier: LogisticRegression, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = data
        predictor = efficient_credal_prediction(classifier)
        logits = predict(predictor, x).logits
        predictor.lower, predictor.upper = compute_efficient_credal_prediction_bounds(
            logits, y, num_classes=3, alpha=0.8
        )

        credal_set = representer(predictor).predict(x)

        assert isinstance(credal_set, ProbabilityIntervalsCredalSet)
        probabilities = classifier.predict_proba(x)
        assert credal_set.lower().shape == probabilities.shape
        assert np.all(credal_set.lower() <= probabilities + 1e-12)
        assert np.all(credal_set.upper() >= probabilities - 1e-12)
        assert np.any(credal_set.upper() - credal_set.lower() > 1e-3)
