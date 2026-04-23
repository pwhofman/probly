"""Torch tests for logit calibration methods."""

from __future__ import annotations

import numpy as np
import pytest

from probly.calibrator import calibrate
from probly.method.calibration import (
    isotonic_regression,
    platt_scaling,
    sklearn_identity_logit_estimator,
    temperature_scaling,
    torch_identity_logit_model,
    vector_scaling,
)
from probly.method.conformal import conformal_lac
from probly.predictor import LogitClassifier, predict, predict_raw

pytest.importorskip("torch")
import torch
from torch import nn
import torch.nn.functional as F

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


def _make_logits_model(out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(2, out_dim))


def _sample_multiclass_logits(seed: int, num_samples: int, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_samples, num_classes, generator=generator)
    probs = torch.softmax(logits, dim=-1)
    labels = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return logits, labels


def _sample_binary_logits(seed: int, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_samples, generator=generator)
    probs = torch.sigmoid(logits)
    labels = torch.bernoulli(probs, generator=generator)
    return logits, labels


def _multiclass_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1).long()).item())


def _binary_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.binary_cross_entropy_with_logits(logits.reshape(-1), labels.reshape(-1).float()).item())


def _binary_nll_from_probs(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs_clipped = torch.clamp(probs.reshape(-1).to(torch.float64), min=1e-7, max=1.0 - 1e-7)
    labels_float = labels.reshape(-1).to(torch.float64)
    return float(
        (-labels_float * torch.log(probs_clipped) - (1.0 - labels_float) * torch.log(1.0 - probs_clipped)).mean().item()
    )


def _binary_ece_from_probs(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    probs_flat = probs.reshape(-1)
    labels_flat = labels.reshape(-1).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=probs_flat.dtype, device=probs_flat.device)
    bucketized = torch.bucketize(probs_flat, edges[1:-1], right=True)

    ece = torch.tensor(0.0, dtype=probs_flat.dtype, device=probs_flat.device)
    for bin_idx in range(n_bins):
        mask = bucketized == bin_idx
        if bool(mask.any()):
            confidence = probs_flat[mask].mean()
            accuracy = labels_flat[mask].mean()
            ece = ece + torch.abs(confidence - accuracy) * mask.float().mean()
    return float(ece.item())


def _assert_torch_sklearn_probabilities_close(
    torch_probs: torch.Tensor,
    sklearn_probs: np.ndarray,
    *,
    mean_abs_tol: float,
    max_abs_tol: float,
) -> None:
    torch_np = torch_probs.detach().cpu().numpy()
    sklearn_np = np.asarray(sklearn_probs, dtype=float)
    np.testing.assert_equal(torch_np.shape, sklearn_np.shape)
    abs_diff = np.abs(torch_np - sklearn_np)
    assert float(np.mean(abs_diff)) <= mean_abs_tol
    assert float(np.max(abs_diff)) <= max_abs_tol


def test_temperature_scaling_requires_calibration_before_forward() -> None:
    """Uncalibrated wrappers fail before returning predictions."""
    model = temperature_scaling(_make_logits_model(3))
    x = torch.randn(8, 2)
    with pytest.raises(ValueError, match="not calibrated"):
        _ = predict_raw(model, x)


def test_temperature_and_platt_scaling_calibrate_and_predict() -> None:
    """Scalar scaling variants calibrate and produce valid categorical predictions."""
    x = torch.randn(64, 2)

    multiclass_model = temperature_scaling(_make_logits_model(3))
    multiclass_labels = torch.randint(0, 3, (64,))
    calibrate(multiclass_model, multiclass_labels, x)
    logits = predict_raw(multiclass_model, x)
    assert logits.shape == (64, 3)
    dist = predict(multiclass_model, x)
    assert torch.allclose(dist.probabilities.sum(dim=-1), torch.ones(64), atol=1e-6)

    binary_model = platt_scaling(_make_logits_model(1))
    binary_labels = torch.randint(0, 2, (64,)).float()
    calibrate(binary_model, binary_labels, x)
    binary_logits = predict_raw(binary_model, x)
    assert binary_logits.shape == (64, 1)


def test_vector_scaling_state_dict_roundtrip_restores_calibration_state() -> None:
    """Vector scaling parameters are serialized via buffers."""
    x_calib = torch.randn(96, 2)
    y_calib = torch.randint(0, 3, (96,))

    model = vector_scaling(_make_logits_model(3), num_classes=3)
    calibrate(model, y_calib, x_calib)
    state_dict = model.state_dict()
    assert "_temperature" in state_dict
    assert "_is_calibrated" in state_dict

    fresh = vector_scaling(_make_logits_model(3), num_classes=3)
    fresh.load_state_dict(state_dict)

    logits = predict_raw(fresh, torch.randn(16, 2))
    assert logits.shape == (16, 3)


def test_isotonic_state_dict_roundtrip_restores_calibration_state() -> None:
    """Isotonic parameters are serialized in torch state dict buffers."""
    x_calib = torch.randn(256)
    probs = torch.sigmoid(x_calib)
    y_calib = torch.bernoulli(probs)

    model = isotonic_regression(torch_identity_logit_model(), predictor_type=LogitClassifier)
    calibrate(model, y_calib, x_calib)
    state_dict = model.state_dict()
    assert "_isotonic_x_knots" in state_dict
    assert "_isotonic_y_knots" in state_dict
    assert "_isotonic_num_knots" in state_dict

    fresh = isotonic_regression(torch_identity_logit_model(), predictor_type=LogitClassifier)
    fresh.load_state_dict(state_dict)

    test_logits = torch.linspace(-2.0, 2.0, 21)
    restored = predict_raw(fresh, test_logits)
    original = predict_raw(model, test_logits)
    torch.testing.assert_close(restored, original)


def test_torch_temperature_scaling_matches_sklearn_on_identity_logits() -> None:
    """Torch and sklearn temperature scaling should produce near-identical calibrated probabilities."""
    true_calib_logits, y_calib = _sample_multiclass_logits(seed=5100, num_samples=7000, num_classes=4)
    true_test_logits, _ = _sample_multiclass_logits(seed=5300, num_samples=4000, num_classes=4)
    x_calib = true_calib_logits * 2.35
    x_test = true_test_logits * 2.35

    torch_wrapper = temperature_scaling(torch_identity_logit_model())
    sklearn_wrapper = temperature_scaling(sklearn_identity_logit_estimator())

    calibrate(torch_wrapper, y_calib, x_calib)
    calibrate(sklearn_wrapper, y_calib.numpy().astype(int), x_calib.numpy())

    torch_probs = torch.softmax(predict_raw(torch_wrapper, x_test), dim=-1)
    sklearn_probs = sklearn_wrapper.predict_proba(x_test.numpy())
    _assert_torch_sklearn_probabilities_close(torch_probs, sklearn_probs, mean_abs_tol=3e-3, max_abs_tol=2e-2)


def test_torch_platt_scaling_matches_sklearn_on_identity_logits() -> None:
    """Torch and sklearn platt scaling should produce near-identical calibrated binary probabilities."""
    true_calib_logits, y_calib = _sample_binary_logits(seed=5500, num_samples=9000)
    true_test_logits, _ = _sample_binary_logits(seed=5700, num_samples=5000)
    x_calib = (true_calib_logits * 2.15 - 0.55).unsqueeze(-1)
    x_test = (true_test_logits * 2.15 - 0.55).unsqueeze(-1)

    torch_wrapper = platt_scaling(torch_identity_logit_model())
    sklearn_wrapper = platt_scaling(sklearn_identity_logit_estimator())

    calibrate(torch_wrapper, y_calib, x_calib)
    calibrate(sklearn_wrapper, y_calib.numpy().astype(int), x_calib.numpy())

    torch_probs = torch.sigmoid(predict_raw(torch_wrapper, x_test).reshape(-1))
    sklearn_probs = sklearn_wrapper.predict_proba(x_test.numpy())[:, 1]
    _assert_torch_sklearn_probabilities_close(torch_probs, sklearn_probs, mean_abs_tol=1e-2, max_abs_tol=4e-2)


def test_torch_vector_scaling_matches_sklearn_on_identity_logits() -> None:
    """Torch and sklearn vector scaling should produce near-identical calibrated probabilities."""
    scales = torch.tensor([2.1, 0.95, 2.3])
    shifts = torch.tensor([0.15, -0.2, 0.35])
    true_calib_logits, y_calib = _sample_multiclass_logits(seed=5900, num_samples=9000, num_classes=3)
    true_test_logits, _ = _sample_multiclass_logits(seed=6100, num_samples=5000, num_classes=3)
    x_calib = true_calib_logits * scales + shifts
    x_test = true_test_logits * scales + shifts

    torch_wrapper = vector_scaling(torch_identity_logit_model(), num_classes=3)
    sklearn_wrapper = vector_scaling(sklearn_identity_logit_estimator(), num_classes=3)

    calibrate(torch_wrapper, y_calib, x_calib)
    calibrate(sklearn_wrapper, y_calib.numpy().astype(int), x_calib.numpy())

    torch_probs = torch.softmax(predict_raw(torch_wrapper, x_test), dim=-1)
    sklearn_probs = sklearn_wrapper.predict_proba(x_test.numpy())
    _assert_torch_sklearn_probabilities_close(torch_probs, sklearn_probs, mean_abs_tol=3e-3, max_abs_tol=2e-2)


def test_torch_isotonic_regression_matches_sklearn_on_identity_logits() -> None:
    """Torch and sklearn isotonic calibration should produce close binary probabilities."""
    true_calib_logits, y_calib = _sample_binary_logits(seed=6300, num_samples=9000)
    true_test_logits, _ = _sample_binary_logits(seed=6500, num_samples=5000)
    x_calib = (5.0 * true_calib_logits - 2.0).unsqueeze(-1)
    x_test = (5.0 * true_test_logits - 2.0).unsqueeze(-1)

    torch_wrapper = isotonic_regression(torch_identity_logit_model(), predictor_type=LogitClassifier)
    sklearn_wrapper = isotonic_regression(sklearn_identity_logit_estimator(), predictor_type=LogitClassifier)

    calibrate(torch_wrapper, y_calib, x_calib)
    calibrate(sklearn_wrapper, y_calib.numpy().astype(int), x_calib.numpy())

    torch_probs = predict(torch_wrapper, x_test).unnormalized_probabilities.reshape(-1)
    sklearn_probs = sklearn_wrapper.predict_proba(x_test.numpy())[:, 1]
    _assert_torch_sklearn_probabilities_close(torch_probs, sklearn_probs, mean_abs_tol=3e-2, max_abs_tol=1.2e-1)


def test_calibration_supports_arbitrary_batch_dims() -> None:
    """Calibration losses flatten arbitrary batch prefixes while preserving output shape."""
    x_multiclass = torch.randn(4, 5, 2)
    y_multiclass = torch.randint(0, 3, (4, 5))
    multiclass = temperature_scaling(_make_logits_model(3))
    calibrate(multiclass, y_multiclass, x_multiclass)
    multiclass_logits = predict_raw(multiclass, x_multiclass)
    assert multiclass_logits.shape == (4, 5, 3)

    class BinaryLogitModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x).squeeze(-1)

    x_binary = torch.randn(3, 6, 2)
    y_binary = torch.randint(0, 2, (3, 6)).float()
    binary = platt_scaling(BinaryLogitModel())
    calibrate(binary, y_binary, x_binary)
    binary_logits = predict_raw(binary, x_binary)
    assert binary_logits.shape == (3, 6)


def test_isotonic_regression_rejects_multiclass_logits() -> None:
    """Isotonic regression currently supports binary logits only."""
    logits = torch.randn(64, 3)
    labels = torch.randint(0, 3, (64,))
    model = isotonic_regression(torch_identity_logit_model(), predictor_type=LogitClassifier)
    with pytest.raises(ValueError, match="binary logits only"):
        calibrate(model, labels, logits)


def test_calibrated_scaling_composes_with_conformal_wrappers() -> None:
    """A calibrated logit-scaling wrapper can be wrapped by conformal prediction."""
    x_scale = torch.randn(64, 2)
    y_scale = torch.randint(0, 3, (64,))
    scaled = temperature_scaling(_make_logits_model(3))
    calibrate(scaled, y_scale, x_scale)

    conformal = conformal_lac(scaled)
    x_conf = torch.randn(64, 2)
    y_conf = torch.randint(0, 3, (64,))
    calibrated_conformal = calibrate(conformal, 0.2, y_conf, x_conf)

    output = predict(calibrated_conformal, torch.randn(8, 2))
    assert hasattr(output, "tensor")


def test_isotonic_regression_improves_binary_nll_and_ece() -> None:
    """Isotonic regression should improve binary NLL and ECE on nonlinearly distorted logits."""
    generator = torch.Generator().manual_seed(77)
    true_calib_logits = torch.randn(9000, generator=generator)
    y_calib = torch.bernoulli(torch.sigmoid(true_calib_logits), generator=generator)
    true_test_logits = torch.randn(7000, generator=generator)
    y_test = torch.bernoulli(torch.sigmoid(true_test_logits), generator=generator)

    distorted_calib_logits = 5.0 * true_calib_logits - 2.0
    distorted_test_logits = 5.0 * true_test_logits - 2.0

    wrapper = isotonic_regression(torch_identity_logit_model(), predictor_type=LogitClassifier)
    raw_probs = torch.sigmoid(distorted_test_logits)
    raw_nll = _binary_nll_from_probs(raw_probs, y_test)
    raw_ece = _binary_ece_from_probs(raw_probs, y_test)

    calibrate(wrapper, y_calib, distorted_calib_logits)
    calibrated_logits = predict_raw(wrapper, distorted_test_logits)
    calibrated_probs = torch.sigmoid(calibrated_logits)
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

    wrapper = temperature_scaling(torch_identity_logit_model())
    raw_nll = _multiclass_nll(x_test, y_test)

    calibrate(wrapper, y_calib, x_calib)
    calibrated_logits = predict_raw(wrapper, x_test)
    calibrated_nll = _multiclass_nll(calibrated_logits, y_test)

    assert calibrated_nll < raw_nll - 0.02
    assert wrapper.temperature is not None
    estimated_temperature = float(wrapper.temperature.reshape(()).item())
    assert estimated_temperature == pytest.approx(scale, rel=0.2, abs=0.2)
    torch.testing.assert_close(torch.argmax(calibrated_logits, dim=-1), torch.argmax(x_test, dim=-1))


@pytest.mark.parametrize(("scale", "shift"), _PLATT_CONFIGS)
def test_platt_scaling_recovers_affine_binary_distortion_and_improves_nll(scale: float, shift: float) -> None:
    """Platt scaling should recover scalar affine binary distortions and improve held-out NLL."""
    seed_offset = round(scale * 100 + shift * 100)
    true_calib_logits, y_calib = _sample_binary_logits(seed=1300 + seed_offset, num_samples=7000)
    true_test_logits, y_test = _sample_binary_logits(seed=1500 + seed_offset, num_samples=5000)
    x_calib = true_calib_logits * scale + shift
    x_test = true_test_logits * scale + shift

    wrapper = platt_scaling(torch_identity_logit_model())
    raw_nll = _binary_nll(x_test, y_test)

    calibrate(wrapper, y_calib, x_calib)
    calibrated_logits = predict_raw(wrapper, x_test)
    calibrated_nll = _binary_nll(calibrated_logits, y_test)

    expected_bias = -shift / scale
    assert calibrated_nll < raw_nll - 0.01
    assert wrapper.temperature is not None
    assert wrapper.bias is not None
    assert torch.isfinite(wrapper.temperature).all()
    assert torch.isfinite(wrapper.bias).all()
    assert float(wrapper.temperature.reshape(()).item()) == pytest.approx(scale, rel=0.25, abs=0.2)
    assert float(wrapper.bias.reshape(()).item()) == pytest.approx(expected_bias, rel=0.3, abs=0.22)
    sorted_indices = torch.argsort(x_test)
    sorted_calibrated = calibrated_logits[sorted_indices]
    assert torch.all(sorted_calibrated[1:] >= sorted_calibrated[:-1] - 1e-6)


@pytest.mark.parametrize(("scale_values", "shift_values"), _VECTOR_CONFIGS)
def test_vector_scaling_recovers_per_class_affine_distortion_and_improves_nll(
    scale_values: tuple[float, float, float],
    shift_values: tuple[float, float, float],
) -> None:
    """Vector scaling should recover per-class affine distortions and improve held-out NLL."""
    scales = torch.tensor(scale_values)
    shifts = torch.tensor(shift_values)

    seed_offset = round(sum(scale_values) * 100 + sum(shift_values) * 100)

    true_calib_logits, y_calib = _sample_multiclass_logits(seed=1800 + seed_offset, num_samples=8000, num_classes=3)
    true_test_logits, y_test = _sample_multiclass_logits(seed=2000 + seed_offset, num_samples=6000, num_classes=3)
    x_calib = true_calib_logits * scales + shifts
    x_test = true_test_logits * scales + shifts

    wrapper = vector_scaling(torch_identity_logit_model(), num_classes=3)
    raw_nll = _multiclass_nll(x_test, y_test)

    calibrate(wrapper, y_calib, x_calib)
    calibrated_logits = predict_raw(wrapper, x_test)
    calibrated_nll = _multiclass_nll(calibrated_logits, y_test)

    expected_bias = -shifts / scales
    assert calibrated_nll < raw_nll - 0.02
    assert wrapper.temperature is not None
    assert wrapper.bias is not None
    assert torch.isfinite(wrapper.temperature).all()
    assert torch.isfinite(wrapper.bias).all()
    assert torch.all(wrapper.temperature > 0)
    torch.testing.assert_close(wrapper.temperature, scales, rtol=0.25, atol=0.2)

    centered_bias = wrapper.bias - wrapper.bias.mean()
    centered_expected_bias = expected_bias - expected_bias.mean()
    torch.testing.assert_close(centered_bias, centered_expected_bias, rtol=0.35, atol=0.28)
