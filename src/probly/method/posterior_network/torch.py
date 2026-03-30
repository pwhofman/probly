"""Torch implementation of the posterior network."""

from __future__ import annotations

import torch
from torch import nn

from probly.layers.torch import RadialNormalizingFlowStack

from ._common import posterior_network_generator


@posterior_network_generator.register(nn.Module)
class TorchPosteriorNetwork(nn.Module):
    """Torch version of Posterior Network."""

    def __init__(
        self,
        encoder: nn.Module,
        dim: int,
        num_classes: int,
        class_counts: list | torch.Tensor | None = None,
        num_flows: int = 6,
    ) -> None:
        """Initialize a posterior network."""
        super().__init__()
        self.encoder = encoder
        self.norm_flow = RadialNormalizingFlowStack(dim=dim, num_classes=num_classes, num_flows=num_flows)
        if class_counts is None:
            class_counts = torch.ones(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of posterior network."""
        x = self.encoder(x)
        log_density = self.norm_flow(x)
        return log_density
