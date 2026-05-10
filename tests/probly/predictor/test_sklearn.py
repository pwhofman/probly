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


# ---------------------------------------------------------------------------
# Tests for the small sklearn predictor helpers (merged from test_sklearn_extras.py).
# ---------------------------------------------------------------------------


class TestCallableAttribute:
    def test_returns_callable(self) -> None:
        from probly.predictor.sklearn import _callable_attribute  # noqa: PLC0415

        class Obj:
            def m(self):
                return 1

        o = Obj()
        result = _callable_attribute(o, "m")
        assert result is not None
        assert callable(result)
        assert result() == 1

    def test_returns_none_when_missing(self) -> None:
        from probly.predictor.sklearn import _callable_attribute  # noqa: PLC0415

        class Obj:
            pass

        assert _callable_attribute(Obj(), "missing") is None

    def test_returns_none_when_not_callable(self) -> None:
        from probly.predictor.sklearn import _callable_attribute  # noqa: PLC0415

        class Obj:
            x = 5

        assert _callable_attribute(Obj(), "x") is None


class TestProbabilitiesToLogits:
    def test_finite_logits_for_valid_probs(self) -> None:
        from probly.predictor.sklearn import _probabilities_to_logits  # noqa: PLC0415

        probs = np.array([[0.1, 0.9]])
        logits = _probabilities_to_logits(probs)
        np.testing.assert_allclose(logits, np.log(probs), atol=1e-10)

    def test_zero_clipped_to_tiny(self) -> None:
        from probly.predictor.sklearn import _probabilities_to_logits  # noqa: PLC0415

        probs = np.array([[0.0, 1.0]])
        logits = _probabilities_to_logits(probs)
        # No -inf despite zero in probs.
        assert np.isfinite(logits).all()


class TestExtractBinaryProbability:
    def test_two_class(self) -> None:
        from probly.predictor.sklearn import _extract_binary_probability  # noqa: PLC0415

        probs = np.array([[0.7, 0.3], [0.4, 0.6]])
        out = _extract_binary_probability(probs)
        np.testing.assert_allclose(out, [0.3, 0.6])

    def test_one_class(self) -> None:
        from probly.predictor.sklearn import _extract_binary_probability  # noqa: PLC0415

        probs = np.array([0.5])
        out = _extract_binary_probability(probs)
        np.testing.assert_allclose(out, [0.5])

    def test_three_class_returned_unchanged(self) -> None:
        from probly.predictor.sklearn import _extract_binary_probability  # noqa: PLC0415

        probs = np.array([[0.2, 0.5, 0.3]])
        out = _extract_binary_probability(probs)
        np.testing.assert_allclose(out, probs)


class TestHasSafeDecisionFunction:
    def test_logistic_regression_yes(self) -> None:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        from probly.predictor.sklearn import _has_safe_decision_function  # noqa: PLC0415

        assert _has_safe_decision_function(LogisticRegression())

    def test_sgd_log_loss_yes(self) -> None:
        from sklearn.linear_model import SGDClassifier  # noqa: PLC0415

        from probly.predictor.sklearn import _has_safe_decision_function  # noqa: PLC0415

        assert _has_safe_decision_function(SGDClassifier(loss="log_loss"))

    def test_sgd_other_loss_no(self) -> None:
        from sklearn.linear_model import SGDClassifier  # noqa: PLC0415

        from probly.predictor.sklearn import _has_safe_decision_function  # noqa: PLC0415

        assert not _has_safe_decision_function(SGDClassifier(loss="hinge"))


class TestSklearnPredict:
    def test_predict_for_logistic_regression_logit(self) -> None:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        from probly.predictor import LogitClassifier, predict_raw  # noqa: PLC0415

        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # noqa: N806
        y = np.array([0, 0, 1, 1])
        clf = LogisticRegression().fit(X, y)
        # Register as LogitClassifier protocol.
        LogitClassifier.register_instance(clf)
        out = predict_raw(clf, X)
        # decision_function for binary classifiers returns 1-D logit array.
        assert out.shape == (4,)

    def test_predict_for_classifier_with_predict_proba(self) -> None:
        from sklearn.tree import DecisionTreeClassifier  # noqa: PLC0415

        from probly.predictor import ProbabilisticClassifier, predict_raw  # noqa: PLC0415

        X = np.array([[0.0], [1.0], [2.0]])  # noqa: N806
        y = np.array([0, 1, 0])
        clf = DecisionTreeClassifier().fit(X, y)
        ProbabilisticClassifier.register_instance(clf)
        out = predict_raw(clf, X)
        # predict_proba returns a 2D array (n_samples, n_classes).
        assert out.shape == (3, 2)

    def test_predict_fallback_to_predict_method(self) -> None:
        # Construct a stub estimator with only ``predict``, no probabilities.
        from sklearn.base import BaseEstimator  # noqa: PLC0415

        from probly.predictor import predict_raw  # noqa: PLC0415

        class StubEstimator(BaseEstimator):
            def predict(self, X):
                return np.zeros(X.shape[0])

        clf = StubEstimator()
        out = predict_raw(clf, np.array([[0.0], [1.0], [2.0]]))
        assert out.shape == (3,)


class TestPrivateLogitHelpers:
    def test_logit_via_decision_function(self) -> None:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        from probly.predictor.sklearn import _sklearn_logit_prediction  # noqa: PLC0415

        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # noqa: N806
        y = np.array([0, 0, 1, 1])
        clf = LogisticRegression().fit(X, y)
        out = _sklearn_logit_prediction(clf, X)
        # decision_function returns a 1-D array for 2-class.
        assert out.ndim == 1

    def test_logit_via_predict_log_proba_when_no_safe_decision(self) -> None:
        from sklearn.tree import DecisionTreeClassifier  # noqa: PLC0415

        from probly.predictor.sklearn import _sklearn_logit_prediction  # noqa: PLC0415

        X = np.array([[0.0], [1.0]])  # noqa: N806
        y = np.array([0, 1])
        clf = DecisionTreeClassifier().fit(X, y)
        out = _sklearn_logit_prediction(clf, X)
        # predict_log_proba returns a 2D array.
        assert out.ndim == 2

    def test_logit_via_predict_proba_when_no_predict_log_proba(self) -> None:
        # Build a stub predictor that has only predict_proba, not predict_log_proba
        # nor a decision_function.
        from probly.predictor.sklearn import _sklearn_logit_prediction  # noqa: PLC0415

        class StubPredictor:
            def predict_proba(self, X):  # noqa: ARG002
                return np.array([[0.4, 0.6]])

        out = _sklearn_logit_prediction(StubPredictor(), np.array([[0.0]]))
        # log of probabilities.
        np.testing.assert_allclose(out, np.log([[0.4, 0.6]]), atol=1e-10)

    def test_logit_no_method_raises(self) -> None:
        from probly.predictor.sklearn import _sklearn_logit_prediction  # noqa: PLC0415

        class StubPredictor:
            pass

        with pytest.raises(NotImplementedError, match="LogitDistributionPredictor"):
            _sklearn_logit_prediction(StubPredictor(), np.array([[0.0]]))


class TestBinaryLogit:
    def test_binary_logit_difference(self) -> None:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        from probly.predictor.sklearn import _sklearn_binary_logit_prediction  # noqa: PLC0415

        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # noqa: N806
        y = np.array([0, 0, 1, 1])
        clf = LogisticRegression().fit(X, y)
        out = _sklearn_binary_logit_prediction(clf, X)
        # For binary classifiers with a 1-D decision function, returns it directly.
        assert out.ndim == 1
