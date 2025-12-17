"""Tests for Isotonic Regression with torch."""

from __future__ import annotations

import pytest
from sklearn.isotonic import IsotonicRegression
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from probly.calibration.isotonic_regression.torch import IsotonicRegressionCalibrator

SetupReturnType = tuple[nn.Sequential, nn.Sequential, Tensor, DataLoader, DataLoader]


@pytest.fixture
def binary_torch_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(10, 2),
    )


@pytest.fixture
def multiclass_torch_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(10, 4),
    )


@pytest.fixture
def setup(binary_torch_model: nn.Sequential, multiclass_torch_model: nn.Sequential) -> SetupReturnType:
    device = torch.device("cpu")
    base_model_multiclass = multiclass_torch_model.to(device)
    base_model_binary = binary_torch_model.to(device)

    inputs = torch.randn(20, 10)
    labels_multiclass = torch.randint(0, 3, (20,))
    labels_binary = torch.randint(0, 2, (20,))

    loader_multiclass = DataLoader(TensorDataset(inputs, labels_multiclass), batch_size=10)
    loader_binary = DataLoader(TensorDataset(inputs, labels_binary), batch_size=10)

    return base_model_multiclass, base_model_binary, inputs, loader_multiclass, loader_binary


def test_fit_binary(setup: SetupReturnType) -> None:
    _, base_model_binary, _, _, calibration_set = setup

    ir_calibrator = IsotonicRegressionCalibrator(base_model_binary, torch.device("cpu"))

    assert len(ir_calibrator.calibrator) == 0

    ir_calibrator.fit(calibration_set)

    assert len(ir_calibrator.calibrator) == 1
    assert isinstance(ir_calibrator.calibrator[0], IsotonicRegression)


def test_predict_binary(setup: SetupReturnType) -> None:
    _, base_model_binary, inputs, _, calibration_set = setup

    ir_calibrator = IsotonicRegressionCalibrator(base_model_binary, torch.device("cpu"))
    ir_calibrator.fit(calibration_set)

    probs_calibrated = ir_calibrator.predict(inputs)

    assert probs_calibrated.shape == (20, 2)
    assert torch.all((probs_calibrated >= 0) & (probs_calibrated <= 1))
    row_sums = probs_calibrated.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))


def test_fit_multiclass(setup: SetupReturnType) -> None:
    base_model_multiclass, _, _, calibration_set, _ = setup

    ir_calibrator = IsotonicRegressionCalibrator(base_model_multiclass, torch.device("cpu"))

    assert len(ir_calibrator.calibrator) == 0

    ir_calibrator.fit(calibration_set)

    assert len(ir_calibrator.calibrator) == 3

    for c in ir_calibrator.calibrator:
        assert isinstance(c, IsotonicRegression)


def test_predict_multiclass(setup: SetupReturnType) -> None:
    base_model_multiclass, _, inputs, calibration_set, _ = setup

    ir_calibrator = IsotonicRegressionCalibrator(base_model_multiclass, torch.device("cpu"))
    ir_calibrator.fit(calibration_set)

    probs_calibrated = ir_calibrator.predict(inputs)

    assert probs_calibrated.shape == (20, 3)
    row_sums = probs_calibrated.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
