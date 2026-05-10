"""Tests for the sklearn calibration helpers.

These tests exercise edge cases and error paths in
``probly.transformation.calibration.sklearn`` that are not covered by the
broader behaviour-level suites under ``tests/probly/method/calibration``.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from probly.transformation.calibration._common import CalibrationMethodConfig
from probly.transformation.calibration.sklearn import (
    SklearnIdentityLogitEstimator,
    SklearnVectorScalingPredictor,
    _extract_calibration_inputs,
    generate_sklearn_scaling_calibrator,
)


def _make_binary_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(40, 2))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    return x, y


class TestExtractCalibrationInputs:
    def test_no_calib_args_raises(self) -> None:
        """Missing calibration args produces a descriptive error."""
        with pytest.raises(ValueError, match="Expected calibration inputs"):
            _extract_calibration_inputs(np.zeros(3), ())

    def test_too_many_args_raises(self) -> None:
        """More than one positional calibration argument is rejected."""
        with pytest.raises(ValueError, match="exactly one calibration input"):
            _extract_calibration_inputs(np.zeros(3), (np.zeros(3), np.zeros(3)))

    def test_returns_swapped_pair(self) -> None:
        """Returns ``(x_calib, y_calib)`` for sklearn's ``fit`` order."""
        y = np.zeros(3)
        x = np.ones((3, 2))
        x_out, y_out = _extract_calibration_inputs(y, (x,))
        assert x_out is x
        assert y_out is y


class TestSklearnIdentityLogitEstimator:
    def test_fit_records_classes(self) -> None:
        """``fit`` populates ``classes_`` from the labels."""
        est = SklearnIdentityLogitEstimator()
        x = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]])
        y = np.array([0, 1, 0])
        out = est.fit(x, y)
        assert out is est
        np.testing.assert_array_equal(est.classes_, np.array([0, 1]))

    def test_fit_rejects_low_dim_logits(self) -> None:
        """1-D logits without an explicit class axis are rejected."""
        est = SklearnIdentityLogitEstimator()
        with pytest.raises(ValueError, match="class axis"):
            est.fit(np.zeros(4), np.zeros(4))

    def test_decision_function_returns_input_unchanged(self) -> None:
        """``decision_function`` is the identity."""
        est = SklearnIdentityLogitEstimator()
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(est.decision_function(x), x)

    def test_decision_function_lazy_classes_default(self) -> None:
        """When fit was not called, ``decision_function`` populates default class indices."""
        est = SklearnIdentityLogitEstimator()
        # ``classes_`` is annotated but not set in __init__ — exercise the lazy branch.
        if hasattr(est, "classes_"):
            del est.classes_
        x = np.array([[0.5, -0.5, 0.0]])
        out = est.decision_function(x)
        np.testing.assert_array_equal(out, x)
        np.testing.assert_array_equal(est.classes_, np.arange(3))

    def test_predict_proba_2d_softmax(self) -> None:
        """``predict_proba`` softmaxes 2-D logits along the last axis."""
        est = SklearnIdentityLogitEstimator()
        logits = np.array([[1.0, 0.0, 0.0]])
        probs = est.predict_proba(logits)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-7)
        # Highest logit corresponds to highest probability
        assert np.argmax(probs, axis=-1)[0] == 0

    def test_predict_proba_1d_uses_sigmoid(self) -> None:
        """1-D logits are interpreted as binary scores via the sigmoid."""
        est = SklearnIdentityLogitEstimator()
        logits = np.array([0.0, 5.0, -5.0])
        probs = est.predict_proba(logits)
        assert probs.shape == (3, 2)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-7)
        # logit 0 -> 0.5/0.5
        np.testing.assert_allclose(probs[0], np.array([0.5, 0.5]), atol=1e-7)

    def test_predict_2d_argmax(self) -> None:
        """``predict`` argmaxes 2-D logits and indexes ``classes_``."""
        est = SklearnIdentityLogitEstimator()
        est.classes_ = np.array([10, 20, 30])
        logits = np.array([[0.1, 0.7, 0.2], [0.9, 0.0, 0.1]])
        out = est.predict(logits)
        np.testing.assert_array_equal(out, np.array([20, 10]))

    def test_predict_1d_threshold(self) -> None:
        """``predict`` thresholds 1-D logits at zero."""
        est = SklearnIdentityLogitEstimator()
        logits = np.array([0.5, -0.5, 0.0, 1e-9])
        out = est.predict(logits)
        np.testing.assert_array_equal(out, np.array([1, 0, 0, 1]))


class _ProbsOnlyEstimator(BaseEstimator):
    """Test estimator returning ``predict_proba`` only (no ``decision_function``)."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.classes_ = np.arange(num_classes)

    def fit(self, _x: np.ndarray, _y: np.ndarray) -> _ProbsOnlyEstimator:
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Convert input to softmax-style probabilities deterministically.
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        last = arr.shape[-1]
        if last >= self.num_classes:
            logits = arr[..., : self.num_classes]
        else:
            logits = np.broadcast_to(arr.mean(axis=-1, keepdims=True), (*arr.shape[:-1], self.num_classes))
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)


class _SingleProbEstimator(BaseEstimator):
    """Estimator that returns a 1-D vector of probabilities."""

    classes_ = np.array([0, 1])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Returns a 1-D positive-class probability vector.
        return 0.5 + 0.0 * np.asarray(x, dtype=float)[:, 0]


class _SingleProbColumnEstimator(BaseEstimator):
    """Estimator that returns a 2-D ``(N, 1)`` probability matrix."""

    classes_ = np.array([0, 1])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 0.5 + 0.0 * np.asarray(x, dtype=float)[:, :1]


class _OpaqueEstimator(BaseEstimator):
    """Estimator without ``decision_function`` or ``predict_proba``."""

    classes_ = np.array([0, 1])


class TestSklearnVectorScalingPredictor:
    def test_uncalibrated_temperature_and_bias_are_none(self) -> None:
        """``temperature`` and ``bias`` are ``None`` until calibration."""
        x_train, y_train = _make_binary_data(0)
        base = LogisticRegression(max_iter=400).fit(x_train, y_train)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2)
        assert wrapper.temperature is None
        assert wrapper.bias is None
        assert wrapper.is_calibrated_ is False

    def test_calibrated_state_exposed(self) -> None:
        """Calibration populates the ``temperature``/``bias`` properties and flag."""
        x, y = _make_binary_data(1)
        base = LogisticRegression(max_iter=400).fit(x, y)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2).fit(x, y)
        assert wrapper.is_calibrated_ is True
        assert wrapper.temperature is not None
        assert wrapper.bias is not None
        np.testing.assert_array_equal(wrapper.temperature.shape, (2,))

    def test_estimator_alias_setter(self) -> None:
        """The sklearn-conventional ``estimator`` alias exposes the wrapped predictor."""
        x_train, y_train = _make_binary_data(2)
        base = LogisticRegression(max_iter=400).fit(x_train, y_train)
        replacement = LogisticRegression(max_iter=400).fit(x_train, y_train)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2)
        assert wrapper.estimator is base

        wrapper.estimator = replacement
        assert wrapper.predictor is replacement
        assert wrapper.estimator is replacement

    def test_predict_uses_classes(self) -> None:
        """``predict`` returns class labels indexed by the wrapped estimator's ``classes_``."""
        x, y = _make_binary_data(3)
        base = LogisticRegression(max_iter=400).fit(x, y)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2).fit(x, y)
        labels = wrapper.predict(x)
        assert labels.shape == (len(x),)
        # Predicted labels must come from the original ``classes_`` set
        assert set(np.unique(labels).tolist()).issubset(set(base.classes_.tolist()))

    def test_predict_without_classes_raises(self) -> None:
        """If ``classes_`` was never populated, ``predict`` errors clearly."""
        x, y = _make_binary_data(4)
        base = LogisticRegression(max_iter=400).fit(x, y)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2).fit(x, y)
        # Force classes_ back to empty to hit the validation branch.
        wrapper.classes_ = np.empty((0,), dtype=np.int64)
        with pytest.raises(ValueError, match="missing classes"):
            wrapper.predict(x)

    def test_extract_logits_via_predict_proba_2_class_matches_decision_function(self) -> None:
        """``_extract_logits`` recovers logits from binary probabilities via the log-ratio trick."""
        x, _y = _make_binary_data(5)
        # Estimator without a decision_function — the log-odds path is exercised.
        base = _ProbsOnlyEstimator(num_classes=2)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2)
        logits = wrapper._extract_logits(x)  # noqa: SLF001
        # We expect logits to have shape (N, 2) with the negative class fixed to zero.
        assert logits.shape[-1] == 2
        np.testing.assert_allclose(logits[..., 0], 0.0)
        # And the positive-class logit should be finite.
        assert np.all(np.isfinite(logits[..., 1]))

    def test_extract_logits_via_predict_proba_multiclass_logs_directly(self) -> None:
        """For >2 classes the wrapper takes ``log(probs)``."""
        x, y = _make_binary_data(6)  # noqa: RUF059
        base = _ProbsOnlyEstimator(num_classes=3)
        # Need 3 columns of input to feed predict_proba -> 3 classes.
        x_with_extra = np.concatenate([x, x[:, :1]], axis=-1)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=3)
        logits = wrapper._extract_logits(x_with_extra)  # noqa: SLF001
        assert logits.shape == (x.shape[0], 3)
        # log probabilities sum to a finite value — they need not be zero.
        assert np.all(np.isfinite(logits))

    def test_extract_logits_handles_1d_predict_proba(self) -> None:
        """A 1-D probability vector is broadened to ``(N, 2)``."""
        x, _ = _make_binary_data(7)
        base = _SingleProbEstimator()
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2)
        logits = wrapper._extract_logits(x)  # noqa: SLF001
        # Both classes should produce identical zero log-odds because probs are 0.5/0.5.
        assert logits.shape == (x.shape[0], 2)

    def test_extract_logits_handles_singleton_class_axis_predict_proba(self) -> None:
        """A ``(N, 1)`` probability matrix is treated as the positive class probability."""
        x, _ = _make_binary_data(8)
        base = _SingleProbColumnEstimator()
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2)
        logits = wrapper._extract_logits(x)  # noqa: SLF001
        assert logits.shape == (x.shape[0], 2)

    def test_extract_logits_decision_function_1d_promotes_to_2d(self) -> None:
        """A 1-D ``decision_function`` is reshaped to ``(N, 2)`` with zero negative-class logit."""

        class OneDimDecisionFn(BaseEstimator):
            classes_ = np.array([0, 1])

            def decision_function(self, x: np.ndarray) -> np.ndarray:
                return np.asarray(x, dtype=float)[:, 0]

        x, _ = _make_binary_data(9)
        base = OneDimDecisionFn()
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2)
        logits = wrapper._extract_logits(x)  # noqa: SLF001
        assert logits.shape == (x.shape[0], 2)
        np.testing.assert_allclose(logits[:, 0], 0.0)

    def test_extract_logits_no_outputs_raises(self) -> None:
        """An estimator without scoring outputs raises an attribute error."""
        wrapper = SklearnVectorScalingPredictor(_OpaqueEstimator(), num_classes=2)
        with pytest.raises(AttributeError, match="neither decision_function nor predict_proba"):
            wrapper._extract_logits(np.zeros((3, 2)))  # noqa: SLF001

    def test_extract_logits_rejects_zero_dim_predict_proba(self) -> None:
        """A predictor returning a 0-D probability scalar is rejected."""

        class ScalarProbEstimator(BaseEstimator):
            classes_ = np.array([0, 1])

            def predict_proba(self, _x: np.ndarray) -> np.ndarray:
                return np.array(0.5)  # 0-D scalar

        wrapper = SklearnVectorScalingPredictor(ScalarProbEstimator(), num_classes=2)
        with pytest.raises(ValueError, match="at least one dimension"):
            wrapper._extract_logits(np.zeros((3, 2)))  # noqa: SLF001

    def test_fit_rejects_low_dim_logits_from_decision_function(self) -> None:
        """If the wrapped estimator's ``decision_function`` returns a 0-D scalar, ``fit`` rejects it."""

        class ScalarDecisionFn(BaseEstimator):
            classes_ = np.array([0, 1])

            def decision_function(self, _x: np.ndarray) -> np.ndarray:
                return np.asarray(0.0)  # 0-D scalar

        wrapper = SklearnVectorScalingPredictor(ScalarDecisionFn(), num_classes=2)
        with pytest.raises(ValueError, match="explicit class axis"):
            wrapper.fit(np.zeros((3, 2)), np.array([0, 1, 0]))

    def test_fit_rejects_one_class_logits(self) -> None:
        """Vector scaling requires more than one class."""

        class OneClassDecisionFn(BaseEstimator):
            classes_ = np.array([0])

            def decision_function(self, x: np.ndarray) -> np.ndarray:
                arr = np.asarray(x, dtype=float)
                # Force a class-axis with size 1.
                return arr[..., :1]

        wrapper = SklearnVectorScalingPredictor(OneClassDecisionFn(), num_classes=None)
        x = np.zeros((5, 2))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="class dimension > 1"):
            wrapper.fit(x, y)

    def test_fit_rejects_num_classes_mismatch(self) -> None:
        """Configured ``num_classes`` must match the logits class dimension."""
        x, y = _make_binary_data(11)
        base = LogisticRegression(max_iter=400).fit(x, y)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=3)
        with pytest.raises(ValueError, match="Expected 3 classes"):
            wrapper.fit(x, y)

    def test_fit_rejects_label_count_mismatch(self) -> None:
        """The number of labels must match the number of logit rows."""
        x, y = _make_binary_data(12)
        base = LogisticRegression(max_iter=400).fit(x, y)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2)
        with pytest.raises(ValueError, match="must match logits batch shape"):
            wrapper.fit(x, y[:-1])

    def test_fit_rejects_estimator_classes_mismatch(self) -> None:
        """``classes_`` exposed by the estimator must match the logits class dimension."""

        class MisalignedClasses(BaseEstimator):
            classes_ = np.array([0, 1, 2])

            def decision_function(self, x: np.ndarray) -> np.ndarray:
                arr = np.asarray(x, dtype=float)
                # Always return 2 logit columns despite advertising 3 classes.
                return arr[..., :2]

        wrapper = SklearnVectorScalingPredictor(MisalignedClasses(), num_classes=2)
        x = np.zeros((5, 2))
        y = np.array([0, 1, 0, 1, 0])
        with pytest.raises(ValueError, match="estimator classes does not match"):
            wrapper.fit(x, y)

    def test_calibrate_delegates_to_fit(self) -> None:
        """``calibrate`` reuses ``fit`` with the probly argument order."""
        x, y = _make_binary_data(13)
        base = LogisticRegression(max_iter=400).fit(x, y)
        wrapper = SklearnVectorScalingPredictor(base, num_classes=2)
        result = wrapper.calibrate(y, x)
        assert result is wrapper
        assert wrapper.is_calibrated_ is True


class TestGenerateSklearnScalingCalibrator:
    def test_temperature_method_requires_recent_sklearn(self) -> None:
        """Temperature scaling raises a clear error on old sklearn versions."""
        config = CalibrationMethodConfig(method="temperature", vector_scale=False, use_bias=False)
        with patch("probly.transformation.calibration.sklearn.sklearn") as mock_sklearn:
            mock_sklearn.__version__ = "1.7.0"
            with pytest.raises(ValueError, match=r"scikit-learn 1\.8\.0 or later"):
                generate_sklearn_scaling_calibrator(LogisticRegression(), config)

    def test_vector_method_returns_vector_predictor(self) -> None:
        """Vector configuration returns the custom ``SklearnVectorScalingPredictor``."""
        config = CalibrationMethodConfig(method="vector", vector_scale=True, use_bias=True, num_classes=3)
        out = generate_sklearn_scaling_calibrator(LogisticRegression(), config)
        assert isinstance(out, SklearnVectorScalingPredictor)
        assert out.num_classes == 3
