"""Torch dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax.nnx import Dropout, Linear, Rngs, Sequential, rnglib

from .common import register

if TYPE_CHECKING:
    from collections.abc import Callable


def prepend_flax_dropout(obj: Callable, p: float, rngs: rnglib.Rngs | rnglib.RngStream | int) -> Sequential:
    """Prepend a Dropout layer before the given layer based on :cite:`galDropoutBayesian2016`."""
    if isinstance(rngs, rnglib.Rngs):
        rngs_metadata = rngs.get_metadata()
        rngs = Rngs(dropout=rngs_metadata)
    if isinstance(rngs, rnglib.RngStream | int):
        rngs = Rngs(dropout=rngs)
    return Sequential(Dropout(p, rngs=rngs), obj)


register(Linear, prepend_flax_dropout)
