"""Temperature scaling extension of base."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from probly.calibration.plattvectortemperature import common

from .torch_base import _LogitScaler


class TorchTemperature(_LogitScaler):
    """Temperature scaling classic (multiclass)."""

    def __init__(self, base: nn.Module) -> None:
        """Initialise a TemperatureScaling object."""
        super().__init__(base, num_classes=-1)
        self.temperature = nn.Parameter(torch.ones(1, device=self.device))

    def _scale_logits(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def _parameters_to_optimize(self) -> list[nn.Parameter]:
        return [self.temperature]

    def _loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        return nn.functional.cross_entropy(logits, labels.long())

@common.register_temperature_factory(nn.Module)
def _(_base: nn.Module) -> type[TorchTemperature]:
    return TorchTemperature
