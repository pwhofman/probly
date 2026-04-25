"""Tests for sklearn-specific predictor dispatch behavior."""

from __future__ import annotations

import numpy as np
import pytest

from probly.predictor import LogitClassifier, predict, predict_raw

pytest.importorskip("sklearn")
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class DummyLogProbabilityEstimator(BaseEstimator):
    """Minimal estimator exposing log-probabilities and margins."""

    def __init__(self, probabilities: np.ndarray) -> None:
        """Initialize fixed class probabilities."""
        self.probabilities = probabilities

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.repeat(self.probabilities[None, :], len(x), axis=0))

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.full((len(x), len(self.probabilities)), 99.0, dtype=float)


class DummyDecisionAndProbabilityEstimator(BaseEstimator):
    """Minimal unknown estimator exposing margins and probabilities."""

    def __init__(self, probabilities: np.ndarray) -> None:
        """Initialize fixed class probabilities."""
        self.probabilities = probabilities

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.full((len(x), len(self.probabilities)), 99.0, dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.probabilities[None, :], len(x), axis=0)


class DummyProbabilityEstimator(BaseEstimator):
    """Minimal estimator exposing probabilities only."""

    def __init__(self, probabilities: np.ndarray) -> None:
        """Initialize fixed class probabilities."""
        self.probabilities = probabilities

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.probabilities[None, :], len(x), axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), int(np.argmax(self.probabilities)), dtype=int)


class DummyDecisionOnlyEstimator(BaseEstimator):
    """Minimal estimator with no meaningful probabilistic API."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=int)


def test_sklearn_logit_predictor_prefers_predict_log_proba() -> None:
    """Log-probabilities are the safest sklearn logit representation."""
    probabilities = np.array([0.2, 0.3, 0.5], dtype=float)
    predictor = DummyLogProbabilityEstimator(probabilities)
    LogitClassifier.register_instance(predictor)

    x = np.zeros((4, 2), dtype=float)
    raw = predict_raw(predictor, x)
    output = predict(predictor, x)

    np.testing.assert_allclose(raw, np.log(np.repeat(probabilities[None, :], len(x), axis=0)))
    np.testing.assert_allclose(output.probabilities, np.repeat(probabilities[None, :], len(x), axis=0))


def test_sklearn_logit_predictor_prefers_known_safe_decision_function() -> None:
    """Known safe sklearn decision functions are treated as raw logits."""
    x, y = make_classification(n_samples=40, n_features=4, n_informative=3, n_redundant=0, random_state=2)
    predictor = LogisticRegression(max_iter=400).fit(x, y)
    LogitClassifier.register_instance(predictor)

    raw = predict_raw(predictor, x[:5])

    np.testing.assert_allclose(raw, predictor.decision_function(x[:5]))


def test_sklearn_logit_predictor_ignores_unknown_decision_function_for_probability_fallback() -> None:
    """Unknown decision functions are not trusted as logits when probabilities are available."""
    probabilities = np.array([0.1, 0.7, 0.2], dtype=float)
    predictor = DummyDecisionAndProbabilityEstimator(probabilities)
    LogitClassifier.register_instance(predictor)

    x = np.zeros((5, 2), dtype=float)
    raw = predict_raw(predictor, x)
    output = predict(predictor, x)

    expected_probabilities = np.repeat(probabilities[None, :], len(x), axis=0)
    np.testing.assert_allclose(raw, np.log(expected_probabilities))
    np.testing.assert_allclose(output.probabilities, expected_probabilities)


def test_sklearn_logit_predictor_falls_back_to_log_probabilities_from_predict_proba() -> None:
    """Probability-only sklearn estimators can still produce logit-equivalent raw predictions."""
    probabilities = np.array([0.1, 0.7, 0.2], dtype=float)
    predictor = DummyProbabilityEstimator(probabilities)
    LogitClassifier.register_instance(predictor)

    x = np.zeros((5, 2), dtype=float)
    raw = predict_raw(predictor, x)
    output = predict(predictor, x)

    expected_probabilities = np.repeat(probabilities[None, :], len(x), axis=0)
    np.testing.assert_allclose(raw, np.log(expected_probabilities))
    np.testing.assert_allclose(output.probabilities, expected_probabilities)


def test_sklearn_logit_predictor_rejects_estimators_without_logit_capable_api() -> None:
    """Manually misannotated sklearn estimators should fail loudly."""
    predictor = DummyDecisionOnlyEstimator()
    LogitClassifier.register_instance(predictor)

    x = np.zeros((2, 2), dtype=float)
    with pytest.raises(NotImplementedError, match="registered as a LogitDistributionPredictor"):
        predict_raw(predictor, x)

    with pytest.raises(NotImplementedError, match="registered as a LogitDistributionPredictor"):
        predict(predictor, x)


def test_sklearn_logit_predictor_rejects_unsafe_svc_margins_without_probabilities() -> None:
    """SVC decision_function outputs are margins, not logits."""
    x, y = make_classification(n_samples=30, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    predictor = SVC(probability=False, random_state=0).fit(x, y)
    LogitClassifier.register_instance(predictor)

    with pytest.raises(NotImplementedError, match="known-safe decision_function"):
        predict_raw(predictor, x[:3])


def test_sklearn_logit_predictor_uses_svc_probabilities_when_available() -> None:
    """SVC probabilities are meaningful even though SVC margins are not logits."""
    x, y = make_classification(n_samples=40, n_features=4, n_informative=3, n_redundant=0, random_state=1)
    predictor = SVC(probability=True, random_state=1).fit(x, y)
    LogitClassifier.register_instance(predictor)

    expected = predictor.predict_proba(x[:4])
    output = predict(predictor, x[:4])

    np.testing.assert_allclose(output.probabilities, expected)


def test_sklearn_probabilistic_classifier_keeps_predict_proba_behavior() -> None:
    """Unannotated sklearn probabilistic classifiers remain categorical predictors."""
    probabilities = np.array([0.8, 0.2], dtype=float)
    predictor = DummyProbabilityEstimator(probabilities)

    x = np.zeros((3, 2), dtype=float)
    raw = predict_raw(predictor, x)
    output = predict(predictor, x)

    expected = np.repeat(probabilities[None, :], len(x), axis=0)
    np.testing.assert_allclose(raw, expected)
    np.testing.assert_allclose(output.probabilities, expected)
