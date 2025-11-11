"""Flax dropconnect implementation."""

from __future__ import annotations

from flax import nnx

from probly.layers.flax import DropConnectLinear

from .common import register


def replace_flax_dropconnect(obj: nnx.Linear, p: float) -> DropConnectLinear:
    """Replace a given layer by a DropConnectLinear layer."""
    return DropConnectLinear(obj, rate=p)


register(nnx.Linear, replace_flax_dropconnect)
