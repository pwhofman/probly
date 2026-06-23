"""Tests for conformalized credal set prediction with PyTorch models."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from probly.calibrator import calibrate
from probly.conformal_scores.total_variation._common import tv_score_func
from probly.method.conformal_credal_set import conformal_total_variation
from probly.predictor import predict
from probly.representation.credal_set.torch import TorchDistanceBasedCredalSet
from probly.utils.quantile._common import calculate_quantile


class DummySoftmaxModel(nn.Module):
    """Model that returns fixed softmax probabilities for testing."""

    def __init__(self, n_classes: int = 3) -> None:
        """Initialize with fixed logits."""
        super().__init__()
        self.n_classes = n_classes
        self._logits = nn.Parameter(torch.tensor([2.0, 1.0, 0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits.softmax(dim=-1).expand(x.shape[0], -1)


class TestConformalCredalSetCalibration:
    """Tests for calibration and quantile storage."""

    @pytest.fixture
    def model(self) -> DummySoftmaxModel:
        return DummySoftmaxModel()

    @pytest.fixture
    def calibration_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        x_calib = torch.randn(20, 4)
        y_calib = torch.randint(0, 3, (20,))
        return x_calib, y_calib

    def test_calibrate_sets_conformal_quantile(
        self, model: DummySoftmaxModel, calibration_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x_calib, y_calib = calibration_data
        predictor = conformal_total_variation(model)
        calibrated = calibrate(predictor, 0.1, y_calib, x_calib)
        assert calibrated.conformal_quantile is not None

    def test_calibrate_quantile_matches_manual(
        self, model: DummySoftmaxModel, calibration_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x_calib, y_calib = calibration_data
        predictor = conformal_total_variation(model)
        calibrated = calibrate(predictor, 0.1, y_calib, x_calib)

        with torch.no_grad():
            preds = model(x_calib)
        manual_scores = tv_score_func(preds, y_calib)
        expected_q = calculate_quantile(manual_scores.numpy(), 0.1)
        assert calibrated.conformal_quantile == pytest.approx(expected_q, abs=1e-6)

    def test_quantile_survives_torch_save_load(
        self, model: DummySoftmaxModel, calibration_data: tuple[torch.Tensor, torch.Tensor], tmp_path: object
    ) -> None:
        x_calib, y_calib = calibration_data
        predictor = conformal_total_variation(model)
        calibrated = calibrate(predictor, 0.1, y_calib, x_calib)
        original_q = calibrated.conformal_quantile

        path = str(tmp_path) + "/model.pt"
        torch.save(calibrated, path)
        loaded = torch.load(path, weights_only=False)
        assert loaded.conformal_quantile == pytest.approx(original_q, abs=1e-10)

    def test_calibrate_with_first_order_targets(self, model: DummySoftmaxModel) -> None:
        """Calibrating with probability-vector targets matches a manual TV-quantile."""
        x_calib = torch.randn(20, 4)
        y_calib_first_order = torch.softmax(torch.randn(20, 3), dim=-1)
        predictor = conformal_total_variation(model)
        calibrated = calibrate(predictor, 0.1, y_calib_first_order, x_calib)

        with torch.no_grad():
            preds = model(x_calib)
        manual_scores = tv_score_func(preds, y_calib_first_order)
        expected_q = calculate_quantile(manual_scores.numpy(), 0.1)
        assert calibrated.conformal_quantile == pytest.approx(expected_q, abs=1e-6)


class TestConformalCredalSetPrediction:
    """Tests for prediction output."""

    @pytest.fixture
    def calibrated_predictor(self) -> nn.Module:
        model = DummySoftmaxModel()
        predictor = conformal_total_variation(model)
        x_calib = torch.randn(20, 4)
        y_calib = torch.randint(0, 3, (20,))
        return calibrate(predictor, 0.1, y_calib, x_calib)

    def test_predict_returns_distance_based_credal_set(self, calibrated_predictor: nn.Module) -> None:
        x_test = torch.randn(5, 4)
        result = predict(calibrated_predictor, x_test)
        assert isinstance(result, TorchDistanceBasedCredalSet)

    def test_predict_radius_equals_quantile(self, calibrated_predictor: nn.Module) -> None:
        x_test = torch.randn(5, 4)
        result = predict(calibrated_predictor, x_test)
        expected_radius = calibrated_predictor.conformal_quantile
        assert torch.allclose(result.radius, torch.tensor(expected_radius))

    def test_predict_nominal_matches_base_model(self, calibrated_predictor: nn.Module) -> None:
        x_test = torch.randn(5, 4)
        result = predict(calibrated_predictor, x_test)
        with torch.no_grad():
            base_model = calibrated_predictor.predictor
            assert isinstance(base_model, nn.Module)
            base_output = base_model(x_test)
        assert torch.allclose(result.nominal.probabilities, base_output, atol=1e-6)

    def test_uncalibrated_predict_raises(self) -> None:
        model = DummySoftmaxModel()
        predictor = conformal_total_variation(model)
        x_test = torch.randn(5, 4)
        with pytest.raises(ValueError, match="not calibrated"):
            predict(predictor, x_test)
