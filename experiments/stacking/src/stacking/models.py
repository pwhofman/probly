"""Opt-in MLP factory used by the v1 composition scripts.

Scripts may use this factory or define their own model class inline; the
playground deliberately does not enforce a model abstraction.
"""

from __future__ import annotations

import torch
from torch import nn

from probly.predictor import LogitClassifier


class StackingMLP(nn.Module, LogitClassifier):
    """Two-or-more-layer MLP returning raw logits.

    Inherits from :class:`probly.predictor.LogitClassifier` so probly's
    calibration and conformal wrappers will accept instances directly.
    """

    def __init__(
        self,
        *,
        in_features: int,
        num_classes: int,
        hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
    ) -> None:
        """Initialise the MLP.

        Args:
            in_features: Number of input features.
            num_classes: Number of output classes (logit dimension).
            hidden: Hidden-layer widths in order; must be non-empty.
            dropout: Dropout probability applied after each hidden ReLU.
                Set to 0.0 to disable dropout entirely.
        """
        super().__init__()
        if not hidden:
            msg = "hidden must contain at least one width"
            raise ValueError(msg)
        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch, num_classes)``."""
        return self.net(x)


def build_mlp(
    *,
    in_features: int,
    num_classes: int,
    hidden: tuple[int, ...] = (128, 128),
    dropout: float = 0.0,
) -> StackingMLP:
    """Construct a :class:`StackingMLP` with the given dimensions.

    Args:
        in_features: Number of input features (e.g. ``Dataset.in_features``).
        num_classes: Number of output classes.
        hidden: Hidden-layer widths.
        dropout: Dropout probability after each hidden ReLU.

    Returns:
        A :class:`StackingMLP` instance.
    """
    return StackingMLP(
        in_features=in_features,
        num_classes=num_classes,
        hidden=hidden,
        dropout=dropout,
    )
