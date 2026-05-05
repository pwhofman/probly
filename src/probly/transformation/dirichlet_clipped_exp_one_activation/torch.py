"""Torch Dirichlet clipped-exp + 1 activation implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.transformation.dirichlet_clipped_exp_one_activation._common import register


class _ClippedExp(nn.Module):
    """Apply ``exp`` to logits clipped to ``[low, high]`` to stabilize the evidence."""

    def __init__(self, low: float = -10.0, high: float = 10.0) -> None:
        """Initialize the clipped-exp module.

        Args:
            low: Lower bound applied to the input before the exponential.
                Defaults to ``-10.0``.
            high: Upper bound applied to the input before the exponential.
                Defaults to ``10.0``.
        """
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x.clamp(self.low, self.high))


class _AddOne(nn.Module):
    """Elementwise +1 module, used to turn evidence into Dirichlet alpha."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def append_activation_torch(obj: nn.Module) -> nn.Sequential:
    """Append clipped exp + 1 so the model outputs Dirichlet alpha = exp(clip(z)) + 1."""
    return nn.Sequential(obj, _ClippedExp(), _AddOne())


register(nn.Module, append_activation_torch)
