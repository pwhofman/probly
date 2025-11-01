"""Torch dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax.nnx import Linear, DropConnectLinear

from .common import register

if TYPE_CHECKING:
    from collections.abc import Callable

def replace_flax_dropconnect(obj: Callable, p: float) -> DropConnectLinear:
    """Replace a given layer by a DropConnectLinear layer."""
    return DropConnectLinear(obj, p=p)


register(Linear, replace_flax_dropconnect)
