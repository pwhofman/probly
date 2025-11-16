"""Flax dropout implementation."""

from __future__ import annotations

from flax import nnx

from probly.layers.flax import DropConnectDense

from .common import register


def replace_flax_dropconnect(obj: nnx.Linear, p: float) -> DropConnectDense:
    """Replace a given layer by a DropConnectDense layer."""
    rngs = nnx.Rngs(0)
    return DropConnectDense(obj, rate=p, rngs=rngs)


register(nnx.Linear, replace_flax_dropconnect)
