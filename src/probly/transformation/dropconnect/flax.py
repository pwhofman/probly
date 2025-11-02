"""Flax NNX DropConnect replacement."""

from __future__ import annotations
from typing import TYPE_CHECKING
from flax import nnx

from probly.layers.flax import DropConnectLinear
from .common import register

if TYPE_CHECKING:
    from flax.nnx import Linear as LinearT  

def replace_flax_dropconnect(layer: nnx.Linear, p: float) -> DropConnectLinear:
    """Ersetzt eine nnx.Linear-Schicht durch DropConnectLinear (Wrapper)."""
    return DropConnectLinear(layer, p=p)


register(nnx.Linear, replace_flax_dropconnect)