"""Platt and vector scaling extension of base."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from probly.calibration.scaling import common

from .torch_base import ScalerTorch


class TorchVector(ScalerTorch):
    """Vector Scaling Implementation with Torch."""

    def __init__(self, base: nn.Module, num_classes: int) -> None:
        """Initialize the Wrapper with a base model and the number of classes.

        Args:
            base: The base model that should be calibrated.
            num_classes: The number of classes the base model was trained on (expects > 1).
        """
        if num_classes <= 1:
            msg = "vector scaling expects num_classes > 1."
            raise ValueError(msg)

        super().__init__(base, num_classes)

        self.w = nn.Parameter(torch.ones(num_classes, device=self.device))
        self.b = nn.Parameter(torch.zeros(num_classes, device=self.device))

    def _scale_logits(self, logits: Tensor) -> Tensor:
        """Scale logits based on the learned parameters."""
        if self.num_classes != logits.shape[-1] or logits.ndim != 2:
            msg = "The given parameter num_classes does not match the actual number of classes."
            raise ValueError(msg)

        return logits * self.w + self.b

    def _parameters_to_optimize(self) -> list[nn.Parameter]:
        """Create a list of all parameters to be optimized."""
        return [self.w, self.b]

    def _loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        """The loss function for vector scaling."""
        return nn.functional.cross_entropy(logits, labels.long())


@common.register_vector_factory(nn.Module)
def _(_base: nn.Module, _num_classes: int) -> type[TorchVector]:
    return TorchVector
