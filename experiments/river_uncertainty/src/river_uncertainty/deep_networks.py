"""PyTorch module definitions for the deep-learning uncertainty experiments."""

from __future__ import annotations

import torch
from torch import nn


class DropoutMLP(nn.Module):
    """MLP with configurable hidden layers and dropout.

    For deep ensemble use, set ``dropout_rate=0.0``.
    For MC Dropout use, set ``dropout_rate > 0`` and run forward passes
    in train mode to keep dropout active.

    Args:
        n_features: Number of input features.
        n_classes: Number of output classes (logits).
        hidden_sizes: Widths of the hidden layers.
        dropout_rate: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h
        layers.append(nn.Linear(in_size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
