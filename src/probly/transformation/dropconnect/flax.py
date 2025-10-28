"""Flax dropout implementation."""

from __future__ import annotations

import flax.nnx as nnx

from probly.layers.flax import DropConnectDense

from .common import register


def replace_flax_dropconnect(obj: nnx.Linear, p: float, rngs: nnx.Rngs) -> DropConnectDense:
    """Replace a given layer by a DropConnectDense layer."""
    return DropConnectDense(obj, p=p, rngs=rngs)


register(nnx.Linear, replace_flax_dropconnect)