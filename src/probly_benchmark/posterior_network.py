"""Benchmarking a posterior network implementation on a simple model."""

from __future__ import annotations

import torch
from torch import nn

from probly.method.posterior_network import posterior_network


# very simple base model for demonstration purposes
class SimpleModel(nn.Module):
    """A simple linear model for demonstration purposes."""

    def __init__(self) -> None:
        """Initialize the SimpleModel."""
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SimpleModel."""
        return self.linear(x)


model = SimpleModel()
print(isinstance(model, nn.Module))  # True
posterior_model = posterior_network(model, dim=10, num_classes=5)
inputs = torch.randn(3, 10)
outputs = posterior_model(inputs)  # ty: ignore
print(outputs)
