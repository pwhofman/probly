from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from probly.calibration.temperature_scaling.torch import TorchTemperature

ReturnTypeFixture = tuple[TorchTemperature, nn.Module, Tensor, DataLoader[tuple[Tensor, ...]]]

@pytest.fixture
def setup_model(torch_custom_model: nn.Sequential) -> ReturnTypeFixture:
    """Set up a dummy model and calibration set."""
    device = torch.device("cpu")
    base = torch_custom_model.to(device)
    temperature_model = TorchTemperature(base, device)

    x = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    calibration_loader = DataLoader(TensorDataset(x, y), batch_size=10)

    return temperature_model, base, x, calibration_loader


def test_forward(setup_model: ReturnTypeFixture) -> None:
    """Check if applying the model returns logits divided by the temperature, within a calculation error tolerance."""
    temperature_model, base, x, _ = setup_model
    logits = base(x)
    scaled_logits = temperature_model(x)

    expected = logits / temperature_model.temperature.item()
    assert torch.allclose(scaled_logits, expected, atol=1e-5)


def test_fit(setup_model: ReturnTypeFixture) -> None:
    """Check if fit updates the temperature."""
    temperature_model, _, _, calibration_loader = setup_model
    initial_temperature = temperature_model.temperature.item()

    temperature_model.fit(calibration_loader, learning_rate=0.1, max_iter=10)

    optimized_temperature = temperature_model.temperature.item()
    assert initial_temperature != optimized_temperature


def test_predict(setup_model: ReturnTypeFixture) -> None:
    """Check if the predictions result in valid probability distributions."""
    temperature_model, _, x, _ = setup_model
    predictions = temperature_model.predict(x)

    assert predictions.shape == (20, 4)
    row_sums = predictions.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
