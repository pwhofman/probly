"""Implementation of temperature scaling in torch."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, device as TorchDevice, nn

from probly.calibration.temperature_scaling import common
from probly.utils.torch import torch_collect_outputs

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class TorchTemperature(nn.Module):
    """Temperature scaling wrapper for torch, inheriting from nn.Module."""

    def __init__(self, base: nn.Module, device: TorchDevice) -> None:
        """Initialize a temperature-scaling module."""
        super().__init__()
        self.base = base
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the model and divide logits by temperature."""
        logits = cast("Tensor", self.base(x))
        return logits / self.temperature.to(logits.device)

    def fit(self, calibration_set: DataLoader, learning_rate: float = 0.01, max_iter: int = 50) -> None:
        """Optimize the temperature using a calibration set."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=learning_rate, max_iter=max_iter)
        logits, labels = torch_collect_outputs(self, calibration_set, self.device)
        labels = labels.long()

        def closure() -> Tensor:
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature.to(logits.device)
            loss = nn.functional.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

    def predict(self, x: Tensor) -> Tensor:
        """Apply the optimized temperature value on a new input."""
        self.eval()
        x = x.to(self.device)  # ensure same device
        with torch.no_grad():
            scaled_logits = self.forward(x)
            return torch.softmax(scaled_logits, dim=1)


@common.register_temperature_factory(nn.Module)
def _(_base: nn.Module, _device: TorchDevice) -> type[TorchTemperature]:
    return TorchTemperature
