"""DropConnect transformation for NNX Dense layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.layers.flax import Dense, DropConnectDense

from .common import register

if TYPE_CHECKING:
    from flax import nnx


def replace_nnx_dropconnect_dense(
    obj: Dense,
    *,
    p: float,
    rngs: nnx.Rngs,
) -> DropConnectDense:
    """Replace an NNX Dense layer with DropConnectDense."""
    return DropConnectDense.from_dense(obj, p=p, rngs=rngs)


# IMPORTANT: register *must* be called with EXACTLY two arguments.
register(Dense, replace_nnx_dropconnect_dense)
