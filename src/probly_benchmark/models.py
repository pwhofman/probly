"""Models for the benchmark."""

from __future__ import annotations

import torch
from torch import nn


class LeNetEncoder(nn.Module):
    """LeNet-5 for MNIST outputting raw logits (no Softmax)."""

    def __init__(self, n_classes: int = 10) -> None:
        """Initialize the model.

        Args:
            n_classes: Number of classes in the dataset
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_classes),
            # No Softmax — Softplus appended by EvidentialClassification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(self.features(x))
