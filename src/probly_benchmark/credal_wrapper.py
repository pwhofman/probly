"""Benchmarking for the credal wrapper method."""

from __future__ import annotations

import torch
from torch import nn

from src.probly.methods import CredalWrapper


class SimpleNN(nn.Module):
    """A simple feedforward neural network for testing the CredalWrapper."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Initialize the neural network."""
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network."""
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = SimpleNN(input_size=10, hidden_size=5, output_size=2)
crewra = CredalWrapper(model, num_members=10)
# create dummy input
input_data = torch.randn(1, 10)
# get predictions from the credal wrapper
predictions = crewra.predict(input_data)
