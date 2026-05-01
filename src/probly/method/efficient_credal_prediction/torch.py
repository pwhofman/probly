"""Torch implementation of efficient credal prediction."""

from __future__ import annotations

import torch
from torch import nn

from ._common import EfficientCredalPredictor, compute_efficient_credal_bounds, efficient_credal_prediction_generator


@efficient_credal_prediction_generator.register(nn.Module)
class TorchEfficientCredalPredictor(nn.Module, EfficientCredalPredictor):
    """Torch module that wraps a softmax-free model and stores credal offsets."""

    predictor: nn.Module

    def __init__(self, predictor: nn.Module) -> None:
        """Initialize the predictor.

        Args:
            predictor: The base model.
        """
        super().__init__()
        self.predictor = predictor
        self.register_buffer("lower", None)
        self.register_buffer("upper", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the predictor."""
        return self.predictor(x)


@compute_efficient_credal_bounds.register(torch.Tensor)
def _(logits: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Compute packed credal interval bounds via 2K logit perturbations."""
    n_classes = logits.shape[-1]
    eye = torch.eye(n_classes, device=logits.device, dtype=logits.dtype)
    perturb_up = upper.unsqueeze(-1) * eye
    perturb_lo = lower.unsqueeze(-1) * eye

    logits_up = logits.unsqueeze(1) + perturb_up.unsqueeze(0)
    logits_lo = logits.unsqueeze(1) + perturb_lo.unsqueeze(0)

    probs_up = torch.softmax(logits_up, dim=-1)
    probs_lo = torch.softmax(logits_lo, dim=-1)

    probs_all = torch.cat([probs_up, probs_lo], dim=1)
    lower_bounds = probs_all.min(dim=1).values
    upper_bounds = probs_all.max(dim=1).values

    return torch.cat([lower_bounds, upper_bounds], dim=-1)
