"""Flax tests for logit calibration methods."""

from __future__ import annotations

import numpy as np
import pytest

from probly.calibrator import calibrate
from probly.method.calibration import (
    flax_identity_logit_model,
    isotonic_regression,
    platt_scaling,
    sklearn_identity_logit_estimator,
    temperature_scaling,
    torch_identity_logit_model,
    vector_scaling,
)
from probly.predictor import BinaryLogitClassifier, predict_raw

pytest.importorskip("jax")
pytest.importorskip("flax")
from flax import nnx
import jax
import jax.numpy as jnp

_TEMPERATURE_CONFIGS = [1.8, 1.95, 2.1, 2.25, 2.4, 2.55, 2.7, 2.85, 3.0, 3.15]

_PLATT_CONFIGS = [
    (1.5, -0.9),
    (1.6, -0.6),
    (1.7, -0.3),
    (1.8, 0.0),
    (1.9, 0.3),
    (2.0, 0.6),
    (2.1, -0.75),
    (2.2, -0.45),
    (2.3, 0.45),
    (2.4, 0.75),
]

_VECTOR_CONFIGS = [
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


def _make_logits_model(out_dim: int, seed: int = 0) -> nnx.Module:
    return nnx.Linear(2, out_dim, rngs=nnx.Rngs(seed))


def _sample_multiclass_logits(seed: int, num_samples: int, num_classes: int) -> tuple[jax.Array, jax.Array]:
    key_logits, key_labels = jax.random.split(jax.random.PRNGKey(seed))
    logits = jax.random.normal(key_logits, (num_samples, num_classes))
    labels = jax.random.categorical(key_labels, logits, axis=-1)
    return logits, labels


def _sample_binary_logits(seed: int, num_samples: int) -> tuple[jax.Array, jax.Array]:
    key_logits, key_labels = jax.random.split(jax.random.PRNGKey(seed))
    logits = jax.random.normal(key_logits, (num_samples,))
    labels = jax.random.bernoulli(key_labels, jax.nn.sigmoid(logits)).astype(jnp.float32)
    return logits, labels


def _multiclass_nll(logits: jax.Array, labels: jax.Array) -> float:
    log_probs = jax.nn.log_softmax(logits.reshape(-1, logits.shape[-1]), axis=-1)
    labels_flat = labels.reshape(-1).astype(jnp.int32)
    picked = jnp.take_along_axis(log_probs, labels_flat[:, None], axis=-1).squeeze(-1)
    return float(-jnp.mean(picked))


def _binary_nll(logits: jax.Array, labels: jax.Array) -> float:
    logits_flat = logits.reshape(-1)
    labels_flat = labels.reshape(-1).astype(logits_flat.dtype)
    log_p1 = jax.nn.log_sigmoid(logits_flat)
    log_p0 = jax.nn.log_sigmoid(-logits_flat)
    return float(-jnp.mean(labels_flat * log_p1 + (1.0 - labels_flat) * log_p0))


def _binary_nll_from_probs(probs: jax.Array, labels: jax.Array) -> float:
    probs_np = np.clip(np.asarray(probs.reshape(-1), dtype=np.float64), 1e-7, 1.0 - 1e-7)
    labels_np = np.asarray(labels.reshape(-1), dtype=np.float64)
    return float(np.mean(-labels_np * np.log(probs_np) - (1.0 - labels_np) * np.log(1.0 - probs_np)))


def _binary_ece_from_probs(probs: jax.Array, labels: jax.Array, n_bins: int = 10) -> float:
    probs_flat = np.asarray(probs.reshape(-1))
    labels_flat = np.asarray(labels.reshape(-1)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucketized = np.searchsorted(edges[1:-1], probs_flat, side="right")

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bucketized == bin_idx
        if np.any(mask):
            confidence = probs_flat[mask].mean()
            accuracy = labels_flat[mask].mean()
            ece += abs(float(confidence - accuracy)) * float(mask.mean())
    return ece


def _assert_flax_sklearn_probabilities_close(
    flax_probs: jax.Array,
    sklearn_probs: np.ndarray,
    *,
    mean_abs_tol: float,
    max_abs_tol: float,
) -> None:
    flax_np = np.asarray(flax_probs, dtype=float)
    sklearn_np = np.asarray(sklearn_probs, dtype=float)
    np.testing.assert_equal(flax_np.shape, sklearn_np.shape)
    abs_diff = np.abs(flax_np - sklearn_np)
    assert float(np.mean(abs_diff)) <= mean_abs_tol
    assert float(np.max(abs_diff)) <= max_abs_tol


def test_temperature_scaling_requires_calibration_before_forward() -> None:
    """Uncalibrated wrappers fail before returning predictions."""
    model = temperature_scaling(_make_logits_model(3))
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 2))
    with pytest.raises(ValueError, match="not calibrated"):
        _ = predict_raw(model, x)


def test_temperature_and_platt_scaling_calibrate_and_predict() -> None:
    """Scalar scaling variants calibrate and produce valid categorical predictions."""
    x = jax.random.normal(jax.random.PRNGKey(1), (64, 2))

    multiclass_model = temperature_scaling(_make_logits_model(3))
    multiclass_labels = jax.random.randint(jax.random.PRNGKey(2), (64,), 0, 3)
    calibrate(multiclass_model, multiclass_labels, x)
    logits = predict_raw(multiclass_model, x)
    assert logits.shape == (64, 3)
    probs = jax.nn.softmax(logits, axis=-1)
    assert jnp.allclose(jnp.sum(probs, axis=-1), jnp.ones(64), atol=1e-5)

    binary_model = platt_scaling(_make_logits_model(1))
    binary_labels = jax.random.bernoulli(jax.random.PRNGKey(3), 0.5, (64,)).astype(jnp.float32)
    calibrate(binary_model, binary_labels, x)
    binary_logits = predict_raw(binary_model, x)
    assert binary_logits.shape == (64, 1)


def test_vector_scaling_state_roundtrips_via_nnx_update() -> None:
    """Vector scaling parameters are serialized via nnx state."""
    x_calib = jax.random.normal(jax.random.PRNGKey(4), (96, 2))
    y_calib = jax.random.randint(jax.random.PRNGKey(5), (96,), 0, 3)

    model = vector_scaling(_make_logits_model(3), num_classes=3)
    calibrate(model, y_calib, x_calib)
    state = nnx.state(model)
    assert "_temperature" in state
    assert "_is_calibrated" in state

    fresh = vector_scaling(_make_logits_model(3), num_classes=3)
    nnx.update(fresh, state)

    logits = predict_raw(fresh, jax.random.normal(jax.random.PRNGKey(6), (16, 2)))
    assert logits.shape == (16, 3)
    assert fresh.is_calibrated is True


def test_isotonic_state_roundtrips_via_nnx_update() -> None:
    """Isotonic parameters are serialized via nnx state."""
    x_calib = jax.random.normal(jax.random.PRNGKey(7), (256,))
    probs = jax.nn.sigmoid(x_calib)
    y_calib = jax.random.bernoulli(jax.random.PRNGKey(8), probs).astype(jnp.float32)

    model = isotonic_regression(flax_identity_logit_model(), predictor_type=BinaryLogitClassifier)
    calibrate(model, y_calib, x_calib)
    state = nnx.state(model)
    assert "_isotonic_x_knots" in state
    assert "_isotonic_y_knots" in state
    assert "_isotonic_num_knots" in state

    fresh = isotonic_regression(flax_identity_logit_model(), predictor_type=BinaryLogitClassifier)
    nnx.update(fresh, state)

    test_logits = jnp.linspace(-2.0, 2.0, 21)
    restored = predict_raw(fresh, test_logits)
    original = predict_raw(model, test_logits)
    assert jnp.allclose(restored, original)


def test_flax_temperature_scaling_matches_sklearn_on_identity_logits() -> None:
    """Flax and sklearn temperature scaling should produce near-identical calibrated probabilities."""
    pytest.importorskip("sklearn", minversion="1.8.0")
    true_calib_logits, y_calib = _sample_multiclass_logits(seed=5100, num_samples=7000, num_classes=4)
    true_test_logits, _ = _sample_multiclass_logits(seed=5300, num_samples=4000, num_classes=4)
    x_calib = true_calib_logits * 2.35
    x_test = true_test_logits * 2.35

    flax_wrapper = temperature_scaling(flax_identity_logit_model())
    sklearn_wrapper = temperature_scaling(sklearn_identity_logit_estimator())

    calibrate(flax_wrapper, y_calib, x_calib)
    calibrate(sklearn_wrapper, np.asarray(y_calib, dtype=int), np.asarray(x_calib))

    flax_probs = jax.nn.softmax(predict_raw(flax_wrapper, x_test), axis=-1)
    sklearn_probs = sklearn_wrapper.predict_proba(np.asarray(x_test))
    _assert_flax_sklearn_probabilities_close(flax_probs, sklearn_probs, mean_abs_tol=3e-3, max_abs_tol=2e-2)


def test_flax_platt_scaling_matches_sklearn_on_identity_logits() -> None:
    """Flax and sklearn platt scaling should produce near-identical calibrated binary probabilities."""
    pytest.importorskip("sklearn")
    true_calib_logits, y_calib = _sample_binary_logits(seed=5500, num_samples=9000)
    true_test_logits, _ = _sample_binary_logits(seed=5700, num_samples=5000)
    x_calib = (true_calib_logits * 2.15 - 0.55)[:, None]
    x_test = (true_test_logits * 2.15 - 0.55)[:, None]

    flax_wrapper = platt_scaling(flax_identity_logit_model())
    sklearn_wrapper = platt_scaling(sklearn_identity_logit_estimator())

    calibrate(flax_wrapper, y_calib, x_calib)
    calibrate(sklearn_wrapper, np.asarray(y_calib, dtype=int), np.asarray(x_calib))

    flax_probs = jax.nn.sigmoid(predict_raw(flax_wrapper, x_test).reshape(-1))
    sklearn_probs = sklearn_wrapper.predict_proba(np.asarray(x_test))[:, 1]
    _assert_flax_sklearn_probabilities_close(flax_probs, sklearn_probs, mean_abs_tol=1e-2, max_abs_tol=4e-2)


def test_flax_vector_scaling_matches_sklearn_on_identity_logits() -> None:
    """Flax and sklearn vector scaling should produce near-identical calibrated probabilities."""
    pytest.importorskip("sklearn")
    scales = jnp.array([2.1, 0.95, 2.3])
    shifts = jnp.array([0.15, -0.2, 0.35])
    true_calib_logits, y_calib = _sample_multiclass_logits(seed=5900, num_samples=9000, num_classes=3)
    true_test_logits, _ = _sample_multiclass_logits(seed=6100, num_samples=5000, num_classes=3)
    x_calib = true_calib_logits * scales + shifts
    x_test = true_test_logits * scales + shifts

    flax_wrapper = vector_scaling(flax_identity_logit_model(), num_classes=3)
    sklearn_wrapper = vector_scaling(sklearn_identity_logit_estimator(), num_classes=3)

    calibrate(flax_wrapper, y_calib, x_calib)
    calibrate(sklearn_wrapper, np.asarray(y_calib, dtype=int), np.asarray(x_calib))

    flax_probs = jax.nn.softmax(predict_raw(flax_wrapper, x_test), axis=-1)
    sklearn_probs = sklearn_wrapper.predict_proba(np.asarray(x_test))
    _assert_flax_sklearn_probabilities_close(flax_probs, sklearn_probs, mean_abs_tol=3e-3, max_abs_tol=2e-2)


def test_flax_isotonic_regression_matches_sklearn_on_identity_logits() -> None:
    """Flax and sklearn isotonic calibration should produce close binary probabilities."""
    pytest.importorskip("sklearn")
    true_calib_logits, y_calib = _sample_binary_logits(seed=6300, num_samples=9000)
    true_test_logits, _ = _sample_binary_logits(seed=6500, num_samples=5000)
    x_calib = (5.0 * true_calib_logits - 2.0)[:, None]
    x_test = (5.0 * true_test_logits - 2.0)[:, None]

    flax_wrapper = isotonic_regression(flax_identity_logit_model(), predictor_type=BinaryLogitClassifier)
    sklearn_wrapper = isotonic_regression(sklearn_identity_logit_estimator(), predictor_type=BinaryLogitClassifier)

    calibrate(flax_wrapper, y_calib, x_calib)
    calibrate(sklearn_wrapper, np.asarray(y_calib, dtype=int), np.asarray(x_calib))

    flax_probs = predict_raw(flax_wrapper, x_test).reshape(-1)
    sklearn_probs = sklearn_wrapper.predict_proba(np.asarray(x_test))[:, 1]
    _assert_flax_sklearn_probabilities_close(flax_probs, sklearn_probs, mean_abs_tol=3e-2, max_abs_tol=1.2e-1)


def test_flax_torch_temperature_scaling_close_on_identity_logits() -> None:
    """Flax and torch temperature scaling should agree closely since they minimize the same NLL."""
    torch = pytest.importorskip("torch")
    true_calib_logits, y_calib = _sample_multiclass_logits(seed=6700, num_samples=7000, num_classes=4)
    true_test_logits, _ = _sample_multiclass_logits(seed=6900, num_samples=4000, num_classes=4)
    x_calib = true_calib_logits * 2.35
    x_test = true_test_logits * 2.35

    flax_wrapper = temperature_scaling(flax_identity_logit_model())
    torch_wrapper = temperature_scaling(torch_identity_logit_model())

    calibrate(flax_wrapper, y_calib, x_calib)
    calibrate(torch_wrapper, torch.as_tensor(np.array(y_calib)), torch.as_tensor(np.array(x_calib)))

    flax_probs = jax.nn.softmax(predict_raw(flax_wrapper, x_test), axis=-1)
    torch_probs = torch.softmax(predict_raw(torch_wrapper, torch.as_tensor(np.array(x_test))), dim=-1)
    _assert_flax_sklearn_probabilities_close(flax_probs, torch_probs.numpy(), mean_abs_tol=3e-3, max_abs_tol=2e-2)


def test_calibration_supports_arbitrary_batch_dims() -> None:
    """Calibration losses flatten arbitrary batch prefixes while preserving output shape."""
    x_multiclass = jax.random.normal(jax.random.PRNGKey(10), (4, 5, 2))
    y_multiclass = jax.random.randint(jax.random.PRNGKey(11), (4, 5), 0, 3)
    multiclass = temperature_scaling(_make_logits_model(3))
    calibrate(multiclass, y_multiclass, x_multiclass)
    multiclass_logits = predict_raw(multiclass, x_multiclass)
    assert multiclass_logits.shape == (4, 5, 3)

    class BinaryLogitModel(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs) -> None:
            super().__init__()
            self.linear = nnx.Linear(2, 1, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            return self.linear(x).squeeze(-1)

    x_binary = jax.random.normal(jax.random.PRNGKey(12), (3, 6, 2))
    y_binary = jax.random.bernoulli(jax.random.PRNGKey(13), 0.5, (3, 6)).astype(jnp.float32)
    binary = platt_scaling(BinaryLogitModel(rngs=nnx.Rngs(0)))
    calibrate(binary, y_binary, x_binary)
    binary_logits = predict_raw(binary, x_binary)
    assert binary_logits.shape == (3, 6)


def test_isotonic_regression_rejects_multiclass_logits() -> None:
    """Isotonic regression currently supports binary logits only."""
    logits = jax.random.normal(jax.random.PRNGKey(14), (64, 3))
    labels = jax.random.randint(jax.random.PRNGKey(15), (64,), 0, 3)
    model = isotonic_regression(flax_identity_logit_model(), predictor_type=BinaryLogitClassifier)
    with pytest.raises(ValueError, match="binary logits only"):
        calibrate(model, labels, logits)


def test_isotonic_regression_improves_binary_nll_and_ece() -> None:
    """Isotonic regression should improve binary NLL and ECE on nonlinearly distorted logits."""
    key = jax.random.PRNGKey(77)
    key_calib, key_calib_labels, key_test, key_test_labels = jax.random.split(key, 4)
    true_calib_logits = jax.random.normal(key_calib, (9000,))
    y_calib = jax.random.bernoulli(key_calib_labels, jax.nn.sigmoid(true_calib_logits)).astype(jnp.float32)
    true_test_logits = jax.random.normal(key_test, (7000,))
    y_test = jax.random.bernoulli(key_test_labels, jax.nn.sigmoid(true_test_logits)).astype(jnp.float32)

    distorted_calib_logits = 5.0 * true_calib_logits - 2.0
    distorted_test_logits = 5.0 * true_test_logits - 2.0

    wrapper = isotonic_regression(flax_identity_logit_model(), predictor_type=BinaryLogitClassifier)
    raw_probs = jax.nn.sigmoid(distorted_test_logits)
    raw_nll = _binary_nll_from_probs(raw_probs, y_test)
    raw_ece = _binary_ece_from_probs(raw_probs, y_test)

    calibrate(wrapper, y_calib, distorted_calib_logits)
    calibrated_probs = predict_raw(wrapper, distorted_test_logits)
    calibrated_nll = _binary_nll_from_probs(calibrated_probs, y_test)
    calibrated_ece = _binary_ece_from_probs(calibrated_probs, y_test)

    assert calibrated_nll < raw_nll - 0.2
    assert calibrated_ece < raw_ece - 0.06


@pytest.mark.parametrize("scale", _TEMPERATURE_CONFIGS)
def test_temperature_scaling_recovers_scalar_and_improves_heldout_nll(scale: float) -> None:
    """Temperature scaling should recover synthetic overconfidence and improve held-out NLL."""
    seed_offset = round(scale * 100)
    true_calib_logits, y_calib = _sample_multiclass_logits(seed=700 + seed_offset, num_samples=7000, num_classes=4)
    true_test_logits, y_test = _sample_multiclass_logits(seed=900 + seed_offset, num_samples=5000, num_classes=4)
    x_calib = true_calib_logits * scale
    x_test = true_test_logits * scale

    wrapper = temperature_scaling(flax_identity_logit_model())
    raw_nll = _multiclass_nll(x_test, y_test)

    calibrate(wrapper, y_calib, x_calib)
    calibrated_logits = predict_raw(wrapper, x_test)
    calibrated_nll = _multiclass_nll(calibrated_logits, y_test)

    assert calibrated_nll < raw_nll - 0.02
    assert wrapper.temperature is not None
    estimated_temperature = float(wrapper.temperature.reshape(()))
    assert estimated_temperature == pytest.approx(scale, rel=0.2, abs=0.2)
    assert jnp.array_equal(jnp.argmax(calibrated_logits, axis=-1), jnp.argmax(x_test, axis=-1))


@pytest.mark.parametrize(("scale", "shift"), _PLATT_CONFIGS)
def test_platt_scaling_recovers_affine_binary_distortion_and_improves_nll(scale: float, shift: float) -> None:
    """Platt scaling should recover scalar affine binary distortions and improve held-out NLL."""
    seed_offset = round(scale * 100 + shift * 100)
    true_calib_logits, y_calib = _sample_binary_logits(seed=1300 + seed_offset, num_samples=7000)
    true_test_logits, y_test = _sample_binary_logits(seed=1500 + seed_offset, num_samples=5000)
    x_calib = true_calib_logits * scale + shift
    x_test = true_test_logits * scale + shift

    wrapper = platt_scaling(flax_identity_logit_model())
    raw_nll = _binary_nll(x_test, y_test)

    calibrate(wrapper, y_calib, x_calib)
    calibrated_logits = predict_raw(wrapper, x_test)
    calibrated_nll = _binary_nll(calibrated_logits, y_test)

    expected_bias = -shift / scale
    assert calibrated_nll < raw_nll - 0.01
    assert wrapper.temperature is not None
    assert wrapper.bias is not None
    assert bool(jnp.isfinite(wrapper.temperature).all())
    assert bool(jnp.isfinite(wrapper.bias).all())
    assert float(wrapper.temperature.reshape(())) == pytest.approx(scale, rel=0.25, abs=0.2)
    assert float(wrapper.bias.reshape(())) == pytest.approx(expected_bias, rel=0.3, abs=0.22)
    sorted_indices = jnp.argsort(x_test)
    sorted_calibrated = calibrated_logits[sorted_indices]
    assert bool(jnp.all(sorted_calibrated[1:] >= sorted_calibrated[:-1] - 1e-6))


@pytest.mark.parametrize(("scale_values", "shift_values"), _VECTOR_CONFIGS)
def test_vector_scaling_recovers_per_class_affine_distortion_and_improves_nll(
    scale_values: tuple[float, float, float],
    shift_values: tuple[float, float, float],
) -> None:
    """Vector scaling should recover per-class affine distortions and improve held-out NLL."""
    scales = jnp.array(scale_values)
    shifts = jnp.array(shift_values)

    seed_offset = round(sum(scale_values) * 100 + sum(shift_values) * 100)

    true_calib_logits, y_calib = _sample_multiclass_logits(seed=1800 + seed_offset, num_samples=8000, num_classes=3)
    true_test_logits, y_test = _sample_multiclass_logits(seed=2000 + seed_offset, num_samples=6000, num_classes=3)
    x_calib = true_calib_logits * scales + shifts
    x_test = true_test_logits * scales + shifts

    wrapper = vector_scaling(flax_identity_logit_model(), num_classes=3)
    raw_nll = _multiclass_nll(x_test, y_test)

    calibrate(wrapper, y_calib, x_calib)
    calibrated_logits = predict_raw(wrapper, x_test)
    calibrated_nll = _multiclass_nll(calibrated_logits, y_test)

    expected_bias = -shifts / scales
    assert calibrated_nll < raw_nll - 0.02
    assert wrapper.temperature is not None
    assert wrapper.bias is not None
    assert bool(jnp.isfinite(wrapper.temperature).all())
    assert bool(jnp.isfinite(wrapper.bias).all())
    assert bool(jnp.all(wrapper.temperature > 0))
    assert jnp.allclose(wrapper.temperature, scales, rtol=0.25, atol=0.2)

    centered_bias = wrapper.bias - wrapper.bias.mean()
    centered_expected_bias = expected_bias - expected_bias.mean()
    assert jnp.allclose(centered_bias, centered_expected_bias, rtol=0.35, atol=0.28)
