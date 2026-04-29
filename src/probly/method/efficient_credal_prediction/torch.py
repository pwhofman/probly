"""Torch implementation of the efficient credal prediction method."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from ._common import efficient_credal_prediction_generator


@efficient_credal_prediction_generator.register(nn.Module)
class TorchEfficientCredalPredictor(nn.Module):
    """Torch nn.Module that wraps a softmax-free model and stores credal bounds."""

    def __init__(self, predictor: nn.Module) -> None:
        """Initialize the predictor.

        Args:
            predictor: The base model.
        """
        super().__init__()
        self.predictor = predictor
        self.register_buffer("lower", None)
        self.register_buffer("upper", None)

    @property
    def lower_bounds(self) -> torch.Tensor:
        """Per-class lower probability bounds."""
        return cast("torch.Tensor", self.lower)

    @property
    def upper_bounds(self) -> torch.Tensor:
        """Per-class upper probability bounds."""
        return cast("torch.Tensor", self.upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the predictor."""
        return self.predictor(x)
