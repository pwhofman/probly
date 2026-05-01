"""Tests for Dirichlet conformalized credal set prediction with PyTorch."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from probly.calibrator import calibrate
from probly.method.conformal_credal_set import conformal_dirichlet_relative_likelihood
from probly.method.prior_network import prior_network
from probly.predictor import LogitClassifier, predict
from probly.representation.credal_set.torch import TorchDirichletLevelSetCredalSet


class DummyLogitModel(nn.Module, LogitClassifier):
    """Model that returns logits for testing."""

    def __init__(self, in_features: int = 4, num_classes: int = 3) -> None:
        """Initialize with a linear layer."""
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
        return self.fc(x)


class TestDirichletConformalCredalSet:
    """Tests for Dirichlet conformal credal set prediction."""

    @pytest.fixture
    def dirichlet_model(self) -> nn.Module:
        """Create a prior network model that outputs Dirichlet parameters."""
        return prior_network(DummyLogitModel())

    @pytest.fixture
    def calibration_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create calibration data."""
        torch.manual_seed(42)
        x_calib = torch.randn(30, 4)
        y_calib = torch.randint(0, 3, (30,))
        return x_calib, y_calib

    def test_calibrate_sets_quantile(
        self, dirichlet_model: nn.Module, calibration_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Calibration should set conformal_quantile."""
        x_calib, y_calib = calibration_data
        predictor = conformal_dirichlet_relative_likelihood(dirichlet_model)
        calibrated = calibrate(predictor, 0.1, y_calib, x_calib)
        assert calibrated.conformal_quantile is not None
        assert 0.0 <= calibrated.conformal_quantile <= 1.0

    def test_predict_returns_dirichlet_level_set(
        self, dirichlet_model: nn.Module, calibration_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Prediction should return a DirichletLevelSetCredalSet."""
        x_calib, y_calib = calibration_data
        predictor = conformal_dirichlet_relative_likelihood(dirichlet_model)
        calibrated = calibrate(predictor, 0.1, y_calib, x_calib)
        x_test = torch.randn(5, 4)
        result = predict(calibrated, x_test)
        assert isinstance(result, TorchDirichletLevelSetCredalSet)

    def test_predict_bounds_valid(
        self, dirichlet_model: nn.Module, calibration_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Lower bounds should be <= upper bounds."""
        x_calib, y_calib = calibration_data
        predictor = conformal_dirichlet_relative_likelihood(dirichlet_model)
        calibrated = calibrate(predictor, 0.1, y_calib, x_calib)
        x_test = torch.randn(5, 4)
        result = predict(calibrated, x_test)
        lower = result.lower()
        upper = result.upper()
        assert torch.all(lower >= 0.0)
        assert torch.all(upper <= 1.0)
        assert torch.all(lower <= upper + 1e-6)

    def test_predict_barycenter_shape(
        self, dirichlet_model: nn.Module, calibration_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Barycenter should have correct shape."""
        x_calib, y_calib = calibration_data
        predictor = conformal_dirichlet_relative_likelihood(dirichlet_model)
        calibrated = calibrate(predictor, 0.1, y_calib, x_calib)
        x_test = torch.randn(5, 4)
        result = predict(calibrated, x_test)
        assert result.barycenter.shape == (5,)
        assert result.num_classes == 3

    def test_uncalibrated_predict_raises(self, dirichlet_model: nn.Module) -> None:
        """Predicting without calibration should raise."""
        predictor = conformal_dirichlet_relative_likelihood(dirichlet_model)
        x_test = torch.randn(5, 4)
        with pytest.raises(ValueError, match="not calibrated"):
            predict(predictor, x_test)

    def test_quantile_survives_serialization(
        self, dirichlet_model: nn.Module, calibration_data: tuple[torch.Tensor, torch.Tensor], tmp_path: object
    ) -> None:
        """Quantile should survive torch save/load."""
        x_calib, y_calib = calibration_data
        predictor = conformal_dirichlet_relative_likelihood(dirichlet_model)
        calibrated = calibrate(predictor, 0.1, y_calib, x_calib)
        original_q = calibrated.conformal_quantile

        path = str(tmp_path) + "/model.pt"
        torch.save(calibrated, path)
        loaded = torch.load(path, weights_only=False)
        assert loaded.conformal_quantile == pytest.approx(original_q, abs=1e-10)
