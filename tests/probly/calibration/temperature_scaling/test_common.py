from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor, nn

from probly.calibration.temperature_scaling import common


class DummyTemperature:
    def __init__(self, base: nn.Module, device: object) -> None:
        """Initialization for."""
        self.base = base
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        output = cast(Tensor, self.base(x))
        return output

@pytest.fixture(autouse=True)
def register_dummy_temperature(torch_custom_model: nn.Sequential) -> None:
    """Registriere DummyTemperature für das Fixture-Modell."""
    @common.register_temperature_factory(type(torch_custom_model))
    def _register(_base: nn.Module, _device: object) -> type[DummyTemperature]:
        return DummyTemperature


def test_temperature_dispatch_returns_instance(torch_custom_model: nn.Sequential) -> None:
    """Temperature.__new__ should return instance of registered class."""
    device = torch.device("cpu")
    temperature_instance = common.Temperature(torch_custom_model, device)

    assert isinstance(temperature_instance, DummyTemperature)
    assert temperature_instance.base is torch_custom_model
    assert temperature_instance.device == device

def test_temperature_dispatch_multiple_instances(torch_custom_model: nn.Sequential) -> None:
    """Dispatcher should handle different instances correctly and not overwrite anything."""
    device = torch.device("cpu")
    instance1 = common.Temperature(torch_custom_model, device)
    instance2 = common.Temperature(torch_custom_model, device)

    assert instance1 is not instance2
    assert instance1.base is torch_custom_model
    assert instance2.base is torch_custom_model



def test_temperature_dispatch_raises_for_unregistered_model() -> None:
    """Dispatcher should throw error a NonImplementedError, when using an unregistered Model."""
    class UnregisteredModel:
        def forward(self, x: Tensor) -> Tensor:
            return x

    model = UnregisteredModel()
    device = torch.device("cpu")

    with pytest.raises(NotImplementedError):
        common.Temperature(model, device)
