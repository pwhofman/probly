"""Implementation for Platt Scaling with Torch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from probly.calibration.scaling import common
from probly.calibration.scaling.torch_base import ScalerTorch


class TorchPlatt(ScalerTorch):
    """Platt scaling Implementation."""

    def __init__(self, base: nn.Module) -> None:
        """Initialize Wrapper with w and biases.

        Args:
            base: The base model that should be calibrated.
        """
        super().__init__(base, num_classes=1)

        self.w = nn.Parameter(torch.ones((), device=self.device))
        self.b = nn.Parameter(torch.zeros((), device=self.device))

    def _scale_logits(self, logits: Tensor) -> Tensor:
        """Scale logits based on the learned parameters."""
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        if logits.shape[-1] != 1 or logits.ndim != 2:
            msg = "platt scaling expects logit shape (N,) or (N,1)."
            raise ValueError(msg)

        return logits * self.w + self.b

    def _parameters_to_optimize(self) -> list[nn.Parameter]:
        """Create a list of all parameters to be optimized."""
        return [self.w, self.b]

    def _loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        """The loss function for platt scaling."""
        labels = labels.float().unsqueeze(1)

        return nn.functional.binary_cross_entropy_with_logits(logits, labels)


@common.register_platt_factory(nn.Module)
def _(_base: nn.Module) -> type[TorchPlatt]:
    return TorchPlatt
