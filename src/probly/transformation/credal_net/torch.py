"""Torch credal net implementation."""

from __future__ import annotations

from torch import nn

from probly.layers.torch import IntSoftmax

from .common import register


def generate_torch_credal_net(model: nn.Module, num_classes: int) -> nn.Module:
    """Build a torch credal net based on :cite:`wang2024credalnet`."""
    last_linear_parent: nn.Module | None = None
    last_linear_name: str | None = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                last_linear_parent = model
                last_linear_name = parts[0]
            else:
                parent: nn.Module = model
                for part in parts[0].split("."):
                    parent = getattr(parent, part)
                last_linear_parent = parent
                last_linear_name = parts[1]

    if last_linear_parent is None or last_linear_name is None:
        msg = "No linear layer found in model"
        raise ValueError(msg)

    last_linear: nn.Linear = getattr(last_linear_parent, last_linear_name)
    setattr(
        last_linear_parent,
        last_linear_name,
        nn.Sequential(
            nn.Linear(last_linear.in_features, 2 * num_classes),
            nn.BatchNorm1d(2 * num_classes),
            IntSoftmax(),
        ),
    )
    return model


register(nn.Module, generate_torch_credal_net)
