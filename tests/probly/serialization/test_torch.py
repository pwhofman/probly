"""Torch serialization behavior for probly models and wrappers."""

from __future__ import annotations

import io

import pytest

from flextype import registry_pickle
from probly.calibrator import calibrate
from probly.layers.torch import NormalInverseGammaLinear
from probly.method.conformal import LACConformalSetPredictor, conformal_lac
from probly.method.dropout import DropoutPredictor, dropout
from probly.method.ensemble import ensemble
from probly.method.evidential import evidential_regression
from probly.predictor import LogitClassifier, RandomPredictor, predict

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


def test_torch_save_with_registry_pickle_preserves_module_type(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Full-object torch serialization with registry pickle should preserve module type."""
    buffer = io.BytesIO()
    torch.save(torch_model_small_2d_2d, buffer, pickle_module=registry_pickle)

    buffer.seek(0)
    restored = torch.load(buffer, pickle_module=registry_pickle, weights_only=False)

    assert type(restored) is type(torch_model_small_2d_2d)


def test_torch_save_with_registry_pickle_preserves_evidential_wrapper_type(
    torch_regression_model_1d: nn.Sequential,
) -> None:
    """Full-object serialization should preserve evidential wrapper layer types."""
    model = evidential_regression(torch_regression_model_1d)
    buffer = io.BytesIO()
    torch.save(model, buffer, pickle_module=registry_pickle)

    buffer.seek(0)
    restored = torch.load(buffer, pickle_module=registry_pickle, weights_only=False)

    assert type(restored) is type(model)
    assert isinstance(restored[-1], NormalInverseGammaLinear)


def test_torch_weights_only_roundtrip_includes_evidential_parameters(torch_regression_model_1d: nn.Sequential) -> None:
    """Weights-only checkpoints should include evidential wrapper parameters."""
    model = evidential_regression(torch_regression_model_1d)
    state_dict = model.state_dict()
    required_keys = {
        "2.gamma",
        "2.nu",
        "2.alpha",
        "2.beta",
        "2.gamma_bias",
        "2.nu_bias",
        "2.alpha_bias",
        "2.beta_bias",
    }
    assert required_keys.issubset(state_dict)

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)

    buffer.seek(0)
    loaded_state_dict = torch.load(buffer, weights_only=True)
    assert required_keys.issubset(loaded_state_dict)

    fresh = evidential_regression(
        nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )
    )
    fresh.load_state_dict(loaded_state_dict)

    fresh_state_dict = fresh.state_dict()
    for key in required_keys:
        assert torch.allclose(loaded_state_dict[key], state_dict[key])
        assert torch.allclose(fresh_state_dict[key], state_dict[key])


def test_torch_weights_only_roundtrip_restores_conformal_calibration_state() -> None:
    """Weights-only checkpoints should restore conformal calibration quantiles."""
    model = conformal_lac(
        nn.Sequential(nn.Linear(2, 3)),
        predictor_type=LogitClassifier,
    )
    x_calib = torch.randn(32, 2)
    y_calib = torch.randint(0, 3, (32,))

    calibrated = calibrate(model, 0.2, y_calib, x_calib)
    assert isinstance(calibrated, LACConformalSetPredictor)
    assert calibrated.conformal_quantile is not None

    state_dict = calibrated.state_dict()
    assert "_conformal_quantile" in state_dict

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)

    buffer.seek(0)
    loaded_state_dict = torch.load(buffer, weights_only=True)

    fresh = conformal_lac(
        nn.Sequential(nn.Linear(2, 3)),
        predictor_type=LogitClassifier,
    )
    fresh.load_state_dict(loaded_state_dict)

    assert fresh.conformal_quantile is not None
    assert fresh.conformal_quantile == pytest.approx(calibrated.conformal_quantile)

    x_test = torch.randn(8, 2)
    original_prediction = predict(calibrated, x_test)
    restored_prediction = predict(fresh, x_test)
    assert torch.equal(restored_prediction.tensor, original_prediction.tensor)


def test_torch_save_with_registry_pickle_preserves_dropout_random_predictor_registration(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    """Dropout wrapper protocol registrations should survive full-object torch serialization."""
    wrapped = dropout(torch_model_small_2d_2d, p=0.2)
    assert isinstance(wrapped, DropoutPredictor)
    assert isinstance(wrapped, RandomPredictor)

    buffer = io.BytesIO()
    torch.save(wrapped, buffer, pickle_module=registry_pickle)

    buffer.seek(0)
    restored = torch.load(buffer, pickle_module=registry_pickle, weights_only=False)

    assert type(restored) is type(wrapped)
    assert isinstance(restored, DropoutPredictor)
    assert isinstance(restored, RandomPredictor)


def test_torch_save_with_registry_pickle_preserves_random_predictor_registration_on_ensemble_members(
    torch_model_small_2d_2d: nn.Sequential,
) -> None:
    """Explicit RandomPredictor registration on torch ensemble members should survive full-object serialization."""
    wrapped = ensemble(
        torch_model_small_2d_2d,
        num_members=3,
        predictor_type=RandomPredictor,
    )
    assert all(isinstance(member, RandomPredictor) for member in wrapped)

    buffer = io.BytesIO()
    torch.save(wrapped, buffer, pickle_module=registry_pickle)

    buffer.seek(0)
    restored = torch.load(buffer, pickle_module=registry_pickle, weights_only=False)

    assert type(restored) is type(wrapped)
    assert len(restored) == len(wrapped)
    assert all(isinstance(member, RandomPredictor) for member in restored)
