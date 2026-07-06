"""Torch tests for Dirichlet calibration."""

from __future__ import annotations

import pytest

from probly.calibrator import calibrate
from probly.method import dirichlet_calibration as dirichlet_calibration_from_top
from probly.method.calibration import torch_identity_logit_model
from probly.method.dirichlet_calibration import dirichlet_calibration
from probly.predictor import predict_raw

pytest.importorskip("torch")
import torch
import torch.nn.functional as F

NUM_CLASSES = 3
NUM_SAMPLES = 512
SHARPENING = 3.0


def _overconfident_logits(
    num_samples: int = NUM_SAMPLES, num_classes: int = NUM_CLASSES
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic overconfident logits and labels for calibration tests."""
    torch.manual_seed(0)
    labels = torch.randint(0, num_classes, (num_samples,))
    base = torch.randn(num_samples, num_classes)
    # Push probability mass toward the true class, then sharpen to overconfidence.
    base[torch.arange(num_samples), labels] += 1.5
    return base * SHARPENING, labels


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels.long()).item())


def test_forward_shape_and_calibrate_returns_logits() -> None:
    """Calibrated output keeps the class axis and a finite range."""
    logits, labels = _overconfident_logits()
    model = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES)
    model.calibrate(labels, logits)
    out = predict_raw(model, logits)
    assert out.shape == logits.shape
    assert torch.isfinite(out).all()


def test_calibration_reduces_nll() -> None:
    """Fitting Dirichlet calibration lowers NLL on overconfident logits."""
    logits, labels = _overconfident_logits()
    model = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES)
    model.calibrate(labels, logits)
    calibrated = predict_raw(model, logits)
    assert _nll(calibrated, labels) < _nll(logits, labels)


def test_strong_off_diagonal_regularisation_shrinks_off_diagonal() -> None:
    """A large reg_lambda drives the off-diagonal weights toward zero."""
    logits, labels = _overconfident_logits()
    weak = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES, reg_lambda=0.0)
    strong = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES, reg_lambda=1e3)
    weak.calibrate(labels, logits)
    strong.calibrate(labels, logits)

    eye = torch.eye(NUM_CLASSES, dtype=torch.bool)
    weak_off = weak.weight[~eye].abs().mean()
    strong_off = strong.weight[~eye].abs().mean()
    assert strong_off < weak_off


def test_generic_calibrate_matches_fit_alias() -> None:
    """The generic calibrate() and the sklearn-style fit() alias agree."""
    logits, labels = _overconfident_logits()
    via_calibrate = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES)
    calibrate(via_calibrate, labels, logits)

    via_fit = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES)
    via_fit.fit(logits, labels)

    assert torch.allclose(via_calibrate.weight, via_fit.weight, atol=1e-5)
    assert torch.allclose(via_calibrate.bias, via_fit.bias, atol=1e-5)


def test_reg_mu_defaults_to_reg_lambda() -> None:
    """When reg_mu is None it inherits reg_lambda."""
    model = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES, reg_lambda=0.25)
    assert model.reg_mu == pytest.approx(0.25)


@pytest.mark.parametrize("num_classes", [None, 1, 0])
def test_invalid_num_classes_raises(num_classes: int | None) -> None:
    """num_classes must be greater than one."""
    with pytest.raises(ValueError, match="num_classes"):
        dirichlet_calibration(torch_identity_logit_model(), num_classes=num_classes)


def test_predict_before_calibrate_raises() -> None:
    """Prediction before calibration is rejected."""
    logits, _ = _overconfident_logits()
    model = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES)
    with pytest.raises(ValueError, match="not calibrated"):
        predict_raw(model, logits)


def test_state_dict_round_trip_reproduces_predictions() -> None:
    """A reloaded calibrator reproduces the fitted predictions."""
    logits, labels = _overconfident_logits()
    model = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES)
    model.calibrate(labels, logits)
    expected = predict_raw(model, logits)

    restored = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES)
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(predict_raw(restored, logits), expected, atol=1e-6)


def test_exposed_from_top_level_method_namespace() -> None:
    """dirichlet_calibration is importable from probly.method."""
    assert dirichlet_calibration_from_top is dirichlet_calibration


def test_mismatched_class_axis_raises() -> None:
    """Logits whose class axis disagrees with num_classes are rejected."""
    logits, labels = _overconfident_logits(num_classes=NUM_CLASSES)
    model = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES + 1)
    with pytest.raises(ValueError, match="class axis"):
        model.calibrate(labels, logits)


def test_forward_preserves_leading_batch_dimensions() -> None:
    """Prediction broadcasts the calibration map over arbitrary leading dimensions."""
    logits, labels = _overconfident_logits()
    model = dirichlet_calibration(torch_identity_logit_model(), num_classes=NUM_CLASSES)
    model.calibrate(labels, logits)

    batched = logits.reshape(4, -1, NUM_CLASSES)
    out = predict_raw(model, batched)
    assert out.shape == batched.shape
    assert torch.allclose(out.reshape(-1, NUM_CLASSES), predict_raw(model, logits), atol=1e-6)
