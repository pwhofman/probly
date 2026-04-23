"""Neural network models used in benchmarks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as tm

from probly_benchmark.resnet import ResNet18


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
            if pretrained:
                msg = "Pretrained weights are not supported for the local ResNet18."
                raise NotImplementedError(msg)
            model = ResNet18()
            model.linear = nn.Linear(model.linear.in_features, num_classes)
        case "resnet18_encoder":
            if pretrained:
                msg = "Pretrained weights are not supported for the local ResNet18."
                raise NotImplementedError(msg)
            model = ResNet18()
            model.linear = nn.Identity()
        case "resnet50":
            model = tm.resnet50(weights="DEFAULT" if pretrained else None)
            if model.fc.out_features != num_classes:
                model.fc = nn.Linear(model.fc.in_features, num_classes)
        case "resnet50_encoder":
            model = tm.resnet50(weights="DEFAULT" if pretrained else None)
            model.fc = nn.Identity()
        case "convnext_tiny":
            model = tm.convnext_tiny(weights="DEFAULT" if pretrained else None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        case "regnet_y_400mf":
            model = tm.regnet_y_400mf(weights="DEFAULT" if pretrained else None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
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


class SmallResidualBlock(nn.Module):
    """Small residual block with two convolutions."""

    def __init__(self, channels: int) -> None:
        """Initialize the block."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity  # residual connection
        return F.relu(out)


class MiniResNet(nn.Module):
    """MiniResNet model."""

    def __init__(self, n_classes: int = 5) -> None:
        """Initialize the model."""
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.block1 = SmallResidualBlock(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.block2 = SmallResidualBlock(32)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.conv1(x))
        x = self.block1(x)

        x = F.relu(self.conv2(x))
        x = self.block2(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
