"""Platt- and Vectorscaling extension of base."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import _LogitScaler


class AffineScaling(_LogitScaler):
    """Platt scaling (binary) or vector scaling (multiclass)."""

    def __init__(self, base: nn.Module, num_classes: int) -> None:
        """Initialise a AffineScaling object."""
        super().__init__(base, num_classes)

        self.w = nn.Parameter(torch.ones(num_classes, device=self.device))
        self.b = nn.Parameter(torch.zeros(num_classes, device=self.device))

    def _scale_logits(self, logits: Tensor) -> Tensor:
        if self.num_classes == 1 and logits.ndim == 1:
            logits = logits.unsqueeze(1)
        return logits * self.w + self.b

    def _parameters_to_optimize(self) -> list[nn.Parameter]:
        return [self.w, self.b]

    def _loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        if self.num_classes == 1:
            labels = labels.float().unsqueeze(1)
            return nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return nn.functional.cross_entropy(logits, labels.long())
