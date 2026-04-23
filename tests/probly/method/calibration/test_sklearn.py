"""sklearn tests for logit calibration methods."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from probly.calibrator import calibrate
from probly.method.calibration import (
    isotonic_regression,
    platt_scaling,
    sklearn_identity_logit_estimator,
    temperature_scaling,
    vector_scaling,
)
from probly.method.calibration.sklearn import SklearnVectorScalingPredictor
from probly.method.conformal import conformal_lac

pytest.importorskip("sklearn")
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

_SKLEARN_TEMPERATURE_CONFIGS = [1.8, 1.95, 2.1, 2.25, 2.4, 2.55, 2.7, 2.85, 3.0, 3.15]

_SKLEARN_VECTOR_CONFIGS = [
    ((1.5, 0.8, 2.0), (0.5, -0.3, 0.7)),
    ((1.6, 0.82, 2.05), (0.45, -0.25, 0.65)),
    ((1.7, 0.84, 2.1), (0.4, -0.2, 0.6)),
    ((1.8, 0.86, 2.15), (0.35, -0.15, 0.55)),
    ((1.9, 0.88, 2.2), (0.3, -0.1, 0.5)),
    ((2.0, 0.9, 2.25), (0.25, -0.05, 0.45)),
    ((2.1, 0.92, 2.3), (0.2, 0.0, 0.4)),
    ((2.2, 0.94, 2.35), (0.15, 0.05, 0.35)),
    ((2.3, 0.96, 2.4), (0.1, 0.1, 0.3)),
    ((2.4, 0.98, 2.45), (0.05, 0.15, 0.25)),
]


def _make_binary_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(120, 2))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    return x, y


def _sample_multiclass_logits(seed: int, num_samples: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(size=(num_samples, num_classes))
    probs = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    uniforms = rng.uniform(size=(num_samples, 1))
    labels = (uniforms > np.cumsum(probs, axis=-1)).sum(axis=-1).astype(int)
    return logits, labels


def _multiclass_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1).astype(int)
    log_probs = flat_logits - logsumexp(flat_logits, axis=-1, keepdims=True)
    return float(-np.mean(log_probs[np.arange(flat_labels.size), flat_labels]))


def test_temperature_platt_and_isotonic_return_builtin_calibrated_classifier_cv() -> None:
    """Temperature, platt, and isotonic use sklearn's builtin calibration estimator."""
    x, y = _make_binary_data(1)

    temperature = temperature_scaling(LogisticRegression(max_iter=400))
    platt = platt_scaling(LogisticRegression(max_iter=400))
    isotonic = isotonic_regression(LogisticRegression(max_iter=400))

    assert isinstance(temperature, CalibratedClassifierCV)
    assert isinstance(platt, CalibratedClassifierCV)
    assert isinstance(isotonic, CalibratedClassifierCV)

    calibrated_temperature = calibrate(temperature, y, x)
    calibrated_platt = calibrate(platt, y, x)
    calibrated_isotonic = calibrate(isotonic, y, x)

    assert calibrated_temperature is temperature
    assert calibrated_platt is platt
    assert calibrated_isotonic is isotonic
    assert temperature.predict_proba(x).shape == (len(x), 2)
    assert platt.predict_proba(x).shape == (len(x), 2)
    assert isotonic.predict_proba(x).shape == (len(x), 2)


def test_vector_scaling_wrapper_uses_fit_for_calibration_and_exposes_estimator_alias() -> None:
    """Custom sklearn vector scaling wrapper follows sklearn's fit pattern."""
    x_train, y_train = _make_binary_data(2)
    x_calib, y_calib = _make_binary_data(3)

    base = LogisticRegression(max_iter=400).fit(x_train, y_train)
    wrapper = vector_scaling(base, num_classes=2)
    assert isinstance(wrapper, SklearnVectorScalingPredictor)
    assert wrapper.estimator is wrapper.predictor
    assert wrapper.estimator is base

    with pytest.raises(ValueError, match="not calibrated"):
        _ = wrapper.predict_proba(x_calib)

    fitted = wrapper.fit(x_calib, y_calib)
    assert fitted is wrapper
    probs = wrapper.predict_proba(x_calib)
    assert probs.shape == (len(x_calib), 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(x_calib)), atol=1e-6)

    calibrated = calibrate(wrapper, y_calib, x_calib)
    assert calibrated is wrapper


def test_sklearn_conformal_fit_behaves_like_calibrate_and_requires_alpha() -> None:
    """Sklearn conformal wrappers calibrate through fit(X, y, *, alpha=...)."""
    x_train, y_train = _make_binary_data(4)
    x_calib, y_calib = _make_binary_data(5)

    base = LogisticRegression(max_iter=400).fit(x_train, y_train)
    predictor = conformal_lac(base)
    assert predictor.estimator is predictor.predictor

    with pytest.raises(ValueError, match="alpha must be provided"):
        predictor.fit(x_calib, y_calib)

    fitted = predictor.fit(x_calib, y_calib, alpha=0.2)
    assert fitted is predictor
    assert predictor.conformal_quantile is not None


def test_sklearn_vector_scaling_supports_arbitrary_batch_prefixes() -> None:
    """Vector scaling accepts logits with multiple batch dimensions."""

    class DummyStructuredEstimator(BaseEstimator):
        def __init__(self, num_classes: int = 4) -> None:
            self.num_classes = num_classes

        def fit(self, x: np.ndarray, _y: np.ndarray) -> DummyStructuredEstimator:
            self.n_features_in_ = x.shape[-1]
            self.classes_ = np.arange(self.num_classes)
            return self

        def decision_function(self, x: np.ndarray) -> np.ndarray:
            return x[..., : self.num_classes]

    rng = np.random.default_rng(12)
    x_calib = rng.normal(size=(3, 5, 6))
    y_calib = rng.integers(0, 4, size=(3, 5))

    base = DummyStructuredEstimator(num_classes=4).fit(x_calib, y_calib)
    wrapper = vector_scaling(base, num_classes=4)
    fitted = wrapper.fit(x_calib, y_calib)
    assert fitted is wrapper

    logits = wrapper.predict_logits(x_calib)
    probs = wrapper.predict_proba(x_calib)
    assert logits.shape == (3, 5, 4)
    assert probs.shape == (3, 5, 4)
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones((3, 5)), atol=1e-6)


@pytest.mark.parametrize(("scale_values", "shift_values"), _SKLEARN_VECTOR_CONFIGS)
def test_sklearn_vector_scaling_improves_heldout_nll_on_synthetic_distorted_logits(
    scale_values: tuple[float, float, float],
    shift_values: tuple[float, float, float],
) -> None:
    """Vector scaling should improve held-out NLL on synthetic per-class affine distortions."""
    scales = np.array(scale_values, dtype=float)
    shifts = np.array(shift_values, dtype=float)

    seed_offset = round(float(np.sum(scales) * 100 + np.sum(shifts) * 100))

    true_calib_logits, y_calib = _sample_multiclass_logits(seed=2300 + seed_offset, num_samples=8000, num_classes=3)
    true_test_logits, y_test = _sample_multiclass_logits(seed=2500 + seed_offset, num_samples=6000, num_classes=3)
    x_calib = true_calib_logits * scales + shifts
    x_test = true_test_logits * scales + shifts

    base = sklearn_identity_logit_estimator()
    wrapper = vector_scaling(base, num_classes=3)

    raw_nll = _multiclass_nll(x_test, y_test)
    wrapper.fit(x_calib, y_calib)
    calibrated_logits = wrapper.predict_logits(x_test)
    calibrated_nll = _multiclass_nll(calibrated_logits, y_test)

    expected_bias = -shifts / scales
    assert calibrated_nll < raw_nll - 0.02
    assert wrapper.temperature is not None
    assert wrapper.bias is not None
    assert np.isfinite(wrapper.temperature).all()
    assert np.isfinite(wrapper.bias).all()
    assert np.all(wrapper.temperature > 0)
    np.testing.assert_allclose(wrapper.temperature, scales, rtol=0.25, atol=0.2)

    centered_bias = wrapper.bias - np.mean(wrapper.bias)
    centered_expected_bias = expected_bias - np.mean(expected_bias)
    np.testing.assert_allclose(centered_bias, centered_expected_bias, rtol=0.35, atol=0.28)


@pytest.mark.parametrize("scale", _SKLEARN_TEMPERATURE_CONFIGS)
def test_sklearn_temperature_scaling_builtin_improves_heldout_nll(scale: float) -> None:
    """Builtin sklearn temperature calibration should improve held-out NLL on overconfident logits."""
    seed_offset = round(scale * 100)
    true_calib_logits, y_calib = _sample_multiclass_logits(seed=2800 + seed_offset, num_samples=5000, num_classes=4)
    true_test_logits, y_test = _sample_multiclass_logits(seed=3000 + seed_offset, num_samples=3500, num_classes=4)
    x_calib = true_calib_logits * scale
    x_test = true_test_logits * scale

    base = sklearn_identity_logit_estimator()
    calibrator = temperature_scaling(base)

    raw_nll = _multiclass_nll(x_test, y_test)
    calibrate(calibrator, y_calib, x_calib)

    probs = np.asarray(calibrator.predict_proba(x_test), dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    calibrated_nll = float(-np.mean(np.log(probs[np.arange(y_test.size), y_test])))
    assert calibrated_nll < raw_nll - 0.015


@pytest.mark.parametrize(("scale", "shift"), [(5.0, -2.0), (6.0, -3.0), (4.0, -2.0)])
def test_sklearn_isotonic_regression_improves_binary_nll_and_ece(scale: float, shift: float) -> None:
    """Builtin sklearn isotonic calibration should improve binary NLL and ECE on distorted logits."""
    seed_offset = round(scale * 100 + shift * 100)
    rng_cal = np.random.default_rng(4100 + seed_offset)
    rng_test = np.random.default_rng(4300 + seed_offset)

    true_calib_logits = rng_cal.normal(size=9000)
    y_calib = rng_cal.binomial(1, 1.0 / (1.0 + np.exp(-true_calib_logits))).astype(int)
    true_test_logits = rng_test.normal(size=7000)
    y_test = rng_test.binomial(1, 1.0 / (1.0 + np.exp(-true_test_logits))).astype(int)

    x_calib = (true_calib_logits * scale + shift).reshape(-1, 1)
    x_test = (true_test_logits * scale + shift).reshape(-1, 1)

    calibrator = isotonic_regression(sklearn_identity_logit_estimator())
    calibrate(calibrator, y_calib, x_calib)

    uncalibrated_probs = 1.0 / (1.0 + np.exp(-x_test[:, 0]))
    calibrated_probs = np.asarray(calibrator.predict_proba(x_test), dtype=float)[:, 1]

    def binary_nll(y_true: np.ndarray, probs: np.ndarray) -> float:
        clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)
        y_float = y_true.astype(float)
        return float(-np.mean(y_float * np.log(clipped) + (1.0 - y_float) * np.log(1.0 - clipped)))

    def binary_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(probs, edges[1:-1], right=True)
        y_float = y_true.astype(float)
        ece = 0.0
        for bin_idx in range(n_bins):
            mask = idx == bin_idx
            if np.any(mask):
                ece += abs(float(np.mean(probs[mask])) - float(np.mean(y_float[mask]))) * float(np.mean(mask))
        return ece

    assert binary_nll(y_test, calibrated_probs) < binary_nll(y_test, uncalibrated_probs) - 0.2
    assert binary_ece(y_test, calibrated_probs) < binary_ece(y_test, uncalibrated_probs) - 0.08
