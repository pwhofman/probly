"""Flax dropconnect implementation."""

from __future__ import annotations

from flax import nnx
from flax.nnx import rnglib

from probly.layers.flax import DropConnectLinear

from .common import register


def replace_flax_dropconnect(
    obj: nnx.Linear, p: float, rngs: rnglib.Rngs | rnglib.RngStream | int
) -> DropConnectLinear:
    """Replace a given layer by a DropConnectLinear layer."""
    if isinstance(rngs, rnglib.Rngs):
        rngs_metadata = rngs.get_metadata()
        rngs = nnx.Rngs(dropconnect=rngs_metadata)
    if isinstance(rngs, rnglib.RngStream | int):
        rngs = nnx.Rngs(dropconnect=rngs)
    return DropConnectLinear(obj, rate=p, rngs=rngs)


register(nnx.Linear, replace_flax_dropconnect)
