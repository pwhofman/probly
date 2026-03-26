"""Flax dropconnect implementation."""

from __future__ import annotations

from flax.nnx import Linear, Rngs, rnglib

from probly.layers.flax import DropConnectLinear

from .common import register


def replace_flax_dropconnect(
    obj: Linear, p: float, rng_collection: str, rngs: rnglib.Rngs | rnglib.RngStream | int
) -> DropConnectLinear:
    """Replace a given layer by a DropConnectLinear layer based on :cite:`mobinyDropConnectEffective2019`."""
    if isinstance(rngs, int):
        rngs = Rngs(rngs)
    return DropConnectLinear(obj, rate=p, rng_collection=rng_collection, rngs=rngs)


register(Linear, replace_flax_dropconnect)
