"""Torch implementation of the efficient credal prediction method."""

from __future__ import annotations

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

    def forward(self, x: torch.Tensor) -> nn.Module:
        """Forward pass through the predictor."""
        return self.predictor(x)
