"""Neural network models used in benchmarks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as tm

from probly.predictor import LogitDistributionPredictor
from probly_benchmark.resnet import ResNet18


def get_encoder_head(name: str, feature_dim: int, num_classes: int) -> nn.Module:
    """Get a classifier head to stack on top of an ``<base>_encoder`` backbone.

    Args:
        name: Head name as it appears in the method config (e.g. ``"linear"``).
        feature_dim: Output feature dim of the encoder (e.g. from ``get_output_dim``).
        num_classes: Number of classes in the dataset.

    Returns:
        A classifier head as a PyTorch module.
    """
    match name:
        case "linear":
            return nn.Linear(feature_dim, num_classes)
        case _:
            msg = f"Encoder head {name} not recognized"
            raise ValueError(msg)


@LogitDistributionPredictor.register_factory
def get_base_model(  # noqa: PLR0912, PLR0915, C901
    name: str,
    num_classes: int,
    pretrained: bool = False,
    in_features: int | None = None,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> nn.Module:
    """Get a base model.

    Args:
        name: Name of the model
        num_classes: Number of classes in the dataset
        pretrained: Whether or not to use pretrained model. Defaults to False.
        in_features: Number of input features for tabular models. Ignored for image models.
            Required for tabular_mlp and tabular_mlp_encoder.
        **kwargs: Additional model-specific arguments.

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
            model.linear = nn.Identity()  # ty: ignore[invalid-assignment]
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
        case "lenet":
            if pretrained:
                msg = "Pretrained weights are not supported for LeNet."
                raise NotImplementedError(msg)
            model = LeNet(n_classes=num_classes)
        case "lenet_encoder":
            if pretrained:
                msg = "Pretrained weights are not supported for LeNet."
                raise NotImplementedError(msg)
            model = LeNet(n_classes=num_classes)
            # Drop final Linear, keeping up to Tanh -> output 84-d
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        case "tabular_mlp":
            if pretrained:
                msg = "Pretrained weights are not supported for TabularMLP."
                raise NotImplementedError(msg)
            if in_features is None:
                msg = "TabularMLP requires 'in_features' kwarg."
                raise ValueError(msg)
            model = TabularMLP(in_features=in_features, n_classes=num_classes)
        case "tabular_mlp_encoder":
            if pretrained:
                msg = "Pretrained weights are not supported for TabularMLP."
                raise NotImplementedError(msg)
            if in_features is None:
                msg = "TabularMLP encoder requires 'in_features' kwarg."
                raise ValueError(msg)
            model = TabularMLP(in_features=in_features, n_classes=num_classes)
            model.lin_out = nn.Identity()  # ty: ignore[invalid-assignment]  # drop classification head, output 1024-d
        case _:
            msg = f"Model {name} not recognized"
            raise ValueError(msg)
    return model


class LeNet(nn.Module):
    """LeNet-5 adapted for 28x28 grayscale input (MNIST).

    Architecture: Conv -> Pool -> Conv -> Pool -> FC -> FC -> FC (logits).
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


class TabularMLP(nn.Module):
    """Two-layer MLP for tabular data.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear.
    """

    def __init__(self, in_features: int, n_classes: int, hidden_dim: int = 1024) -> None:
        """Initialize the model.

        Args:
            in_features: Number of input features.
            n_classes: Number of output classes.
            hidden_dim: Number of hidden units in each layer.
        """
        super().__init__()
        self.lin_one = nn.Linear(in_features, hidden_dim)
        self.lin_two = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.lin_one(x))
        x = F.relu(self.lin_two(x))
        return self.lin_out(x)
