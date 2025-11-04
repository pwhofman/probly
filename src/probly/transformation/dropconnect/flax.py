"""Torch dropout implementation."""

from __future__ import annotations

from flax import nnx

from probly.layers.my_flax import DropConnectLinear

from .common import register


def replace_flax_dropconnect(obj: nnx.Linear, p: float) -> DropConnectLinear:
    """Replace a given nnx.layer by a DropConnectLinear layer."""
    return DropConnectLinear(obj, p=p)


register(nnx.Linear, replace_flax_dropconnect)
