"""Torch evidential classification implementation."""

from __future__ import annotations

from torch import nn

from probly.transformation.evidential.classification.common import register


def append_activation_torch(obj: nn.Module) -> nn.Sequential:
    """Register a base model that the activation function will be appended to."""
    return nn.Sequential(obj, nn.Softplus())


register(nn.Module, append_activation_torch)
