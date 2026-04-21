"""PyTorch conformal predictor wrappers."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch
from torch import nn

from ._common import (
    _ConformalPredictorBase,
    conformal_generator,
)

if TYPE_CHECKING:
    from probly.conformal_scores import NonConformityScore


@conformal_generator.register(nn.Module)
class TorchConformalSetPredictor[**In, Out](_ConformalPredictorBase[In, Out], nn.Module, ABC):
    """Base torch conformal wrapper forwarding ``forward``."""

    predictor: nn.Module

    def __init__(self, predictor: nn.Module, non_conformity_score: NonConformityScore[Out, torch.Tensor]) -> None:
        """Initialize the torch conformal wrapper."""
        super().__init__(predictor, non_conformity_score)

    def forward(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Forward to the wrapped model."""
        return self.predictor(*args, **kwargs)
