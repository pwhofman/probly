"""Temperature scaling extension of base."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from probly.calibration.scaling import common

from .torch_base import ScalerTorch


class TorchTemperature(ScalerTorch):
    """Temperature scaling classic (multiclass)."""

    def __init__(self, base: nn.Module) -> None:
        """Initialize the Wrapper with a base model.

        Args:
            base: The base model that should be calibrated.
        """
        super().__init__(base, num_classes=-1)
        self.temperature = nn.Parameter(torch.ones(1, device=self.device))

    def _scale_logits(self, logits: Tensor) -> Tensor:
        """Scale logits based on the learned parameters."""
        return logits / self.temperature

    def _parameters_to_optimize(self) -> list[nn.Parameter]:
        """Create a list of all parameters to be optimized."""
        return [self.temperature]

    def _loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        """The loss function for temperature scaling."""
        return nn.functional.cross_entropy(logits, labels.long())


@common.register_temperature_factory(nn.Module)
def _(_base: nn.Module) -> type[TorchTemperature]:
    return TorchTemperature
