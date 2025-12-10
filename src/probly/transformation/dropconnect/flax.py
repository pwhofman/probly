"""Flax dropconnect implementation."""

from __future__ import annotations

from flax import nnx

from probly.layers.flax import DropConnectLinear

from .common import register


def replace_flax_dropconnect(obj: nnx.Linear, p: float) -> DropConnectLinear:
    """Replace a Flax Linear layer with DropConnectLinear."""
    # Falls bereits ein DropConnectLinear: NICHT nochmal wrappen, nur p updaten
    if isinstance(obj, DropConnectLinear):  # Geändert: nnx.DropConnectLinear -> DropConnectLinear
        obj.p = float(p)
        return obj

    # Normale Linear-Schicht: durch DropConnectLinear ersetzen
    return DropConnectLinear(obj, p=p)


# Register for Flax Linear layers
register(nnx.Linear, replace_flax_dropconnect)
