"""Torch dropconnect implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

from .common import register

if TYPE_CHECKING:
    from probly.layers.torch import DropConnectLinear

    def replace_flax_dropconnect(obj: nnx.Linear, p: float) -> DropConnectLinear:
        """Replace a given nnx.layer by a DropConnectLinear layer."""
        return DropConnectLinear(obj, p=p)


register(nnx.Linear, replace_flax_dropconnect)
