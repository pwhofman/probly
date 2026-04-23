"""Torch implementation of the efficient credal prediction method."""

from __future__ import annotations

import torch
from torch import nn

from probly.method.efficient_credal_prediction import EfficientCredalPredictor

from ._common import efficient_credal_prediction_generator


@efficient_credal_prediction_generator.register(nn.Module)
class TorchEfficientCredalPredictor(nn.Module, EfficientCredalPredictor):
    """Torch nn.Module that wraps a softmax-free model and stores credal bounds."""

    def __init__(self, predictor: nn.Module, num_classes: int) -> None:
        """Initialize the predictor.

        Args:
            predictor: The base model.
            num_classes: Number of classes.
        """
        super().__init__()
        self.predictor = predictor
        self.register_buffer("lower", torch.zeros(num_classes, dtype=torch.float))
        self.register_buffer("upper", torch.zeros(num_classes, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> nn.Module:
        """Forward pass through the predictor."""
        return self.predictor(x)
