"""Torch dropout implementation."""

from __future__ import annotations

from torch import nn

from probly.transformation.layers import DropConnectLinear

from .common import register


def replace_torch_dropconnect(obj: nn.Module, p: float) -> DropConnectLinear:
    """Replace a given layer by a DropConnectLinear layer."""
    return DropConnectLinear(obj, p=p)


register(nn.Linear, replace_torch_dropconnect)
