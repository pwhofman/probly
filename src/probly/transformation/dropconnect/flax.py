"""Flax dropconnect implementation."""

from __future__ import annotations

from flax.nnx import Linear, Rngs, rnglib

from probly.layers.flax import DropConnectLinear

from .common import register


def replace_flax_dropconnect(obj: Linear, p: float, rngs: rnglib.Rngs | rnglib.RngStream | int) -> DropConnectLinear:
    """Replace a given layer by a DropConnectLinear layer based on :cite:`mobinyDropConnectEffective2019`."""
    if isinstance(rngs, rnglib.Rngs):
        rngs_metadata = rngs.get_metadata()
        rngs = Rngs(dropconnect=rngs_metadata)
    if isinstance(rngs, rnglib.RngStream | int):
        rngs = Rngs(dropconnect=rngs)
    return DropConnectLinear(obj, rate=p, rngs=rngs)


register(Linear, replace_flax_dropconnect)
