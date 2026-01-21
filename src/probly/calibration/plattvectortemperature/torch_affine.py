"""Platt and vector scaling extension of base."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from probly.calibration.plattvectortemperature import common

from .torch_base import _LogitScaler


class TorchAffine(_LogitScaler):
    """Platt scaling (binary) or vector scaling (multiclass)."""

    def __init__(self, base: nn.Module, num_classes: int) -> None:
        """Initialise an AffineScaling object."""
        super().__init__(base, num_classes)

        self.w = nn.Parameter(torch.ones(num_classes, device=self.device))
        self.b = nn.Parameter(torch.zeros(num_classes, device=self.device))

    def _scale_logits(self, logits: Tensor) -> Tensor:
        if self.num_classes != logits.shape[1]:
            msg="The given parameter num_classes does not match the actual number of classes."
            raise ValueError(msg)
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


@common.register_affine_factory(nn.Module)
def _(_base: nn.Module, _num_classes: int) -> type[TorchAffine]:
    return TorchAffine
