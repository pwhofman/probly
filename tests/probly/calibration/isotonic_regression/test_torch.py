"""Tests for Isotonic Regression with torch."""

from __future__ import annotations

from sklearn.isotonic import IsotonicRegression
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from probly.calibration.isotonic_regression.torch import IsotonicRegressionCalibrator

SetupReturnType = tuple[nn.Sequential, nn.Sequential, Tensor, DataLoader, DataLoader]


def test_fit_binary(setup: SetupReturnType) -> None:
    _, base_model_binary, _, _, calibration_set = setup

    ir_calibrator = IsotonicRegressionCalibrator(base_model_binary, False)

    assert len(ir_calibrator.calibrator) == 0

    ir_calibrator.fit(calibration_set)

    assert len(ir_calibrator.calibrator) == 1
    assert isinstance(ir_calibrator.calibrator[0], IsotonicRegression)


def test_predict_binary(setup: SetupReturnType) -> None:
    _, base_model_binary, inputs, _, calibration_set = setup

    ir_calibrator = IsotonicRegressionCalibrator(base_model_binary, False)
    ir_calibrator.fit(calibration_set)

    probs_calibrated = ir_calibrator.predict(inputs)

    assert probs_calibrated.shape == (20, 2)
    assert torch.all((probs_calibrated >= 0) & (probs_calibrated <= 1))
    row_sums = probs_calibrated.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))


def test_fit_multiclass(setup: SetupReturnType) -> None:
    base_model_multiclass, _, _, calibration_set, _ = setup

    ir_calibrator = IsotonicRegressionCalibrator(base_model_multiclass, False)

    assert len(ir_calibrator.calibrator) == 0

    ir_calibrator.fit(calibration_set)

    assert len(ir_calibrator.calibrator) == 3

    for c in ir_calibrator.calibrator:
        assert isinstance(c, IsotonicRegression)


def test_predict_multiclass(setup: SetupReturnType) -> None:
    base_model_multiclass, _, inputs, calibration_set, _ = setup

    ir_calibrator = IsotonicRegressionCalibrator(base_model_multiclass, False)
    ir_calibrator.fit(calibration_set)

    probs_calibrated = ir_calibrator.predict(inputs)

    assert probs_calibrated.shape == (20, 3)
    row_sums = probs_calibrated.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
