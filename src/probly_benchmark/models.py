"""Neural network models used in benchmarks."""

from __future__ import annotations

import torch
from torch import nn
import torchvision.models as tm


def get_base_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Get a base model.

    Args:
        name: Name of the model
        num_classes: Number of classes in the dataset
        pretrained: Whether or not to use pretrained model. Defaults to False.

    Returns:
        The base model as a PyTorch module.
    """
    match name:
        case "resnet18":
            model = tm.resnet18(weights="DEFAULT" if pretrained else None)
            model.fc = nn.Linear(512, num_classes)
        case "resnet18_encoder":
            model = tm.resnet18(weights="DEFAULT" if pretrained else None)
            model.fc = nn.Identity()
        case "resnet50":
            model = tm.resnet50(weights="DEFAULT" if pretrained else None)
            model.fc = nn.Linear(2048, num_classes)
        case "resnet50_encoder":
            model = tm.resnet50(weights="DEFAULT" if pretrained else None)
            model.fc = nn.Identity()
        case _:
            msg = f"Model {name} not recognized"
            raise ValueError(msg)
    return model


class LeNet(nn.Module):
    """LeNet-5 adapted for 28x28 grayscale input (MNIST).

    Architecture: Conv -> Pool -> Conv -> Pool -> FC -> FC -> FC -> Softmax.
    """

    def __init__(self, n_classes: int = 10) -> None:
        """Initialize the model.

        Args:
            n_classes: Number of classes in the dataset
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),  # (1,28,28) -> (6,24,24)
            nn.Tanh(),
            nn.AvgPool2d(2),  # -> (6,12,12)
            nn.Conv2d(6, 16, kernel_size=5),  # -> (16,8,8)
            nn.Tanh(),
            nn.AvgPool2d(2),  # -> (16,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> nn.Module:
        """Forward pass."""
        return self.classifier(self.features(x))
