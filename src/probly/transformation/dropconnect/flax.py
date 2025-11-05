"""Flax dropconnect implementation."""

from __future__ import annotations

import contextlib

from flax import nnx

from probly.layers.flax import DropConnectLinear

from .common import register


def replace_flax_dropconnect(obj: nnx.Linear, p: float) -> DropConnectLinear:
    """Replace a given layer by a DropConnectLinear layer."""
    dc_layer = DropConnectLinear(obj, p=p)

    # Remove the reference to the old base layer
    if hasattr(dc_layer, "base_layer"):
        with contextlib.suppress(Exception):
            delattr(dc_layer, "base_layer")

    return dc_layer


register(nnx.Linear, replace_flax_dropconnect)
