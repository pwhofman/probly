"""Torch dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax.nnx import Dropout, Linear, Rngs, Sequential, rnglib

from .common import register

if TYPE_CHECKING:
    from collections.abc import Callable


def prepend_flax_dropout(
    obj: Callable, p: float, rng_collection: str = "dropout", rngs: rnglib.Rngs | rnglib.RngStream | int = 1
) -> Sequential:
    """Prepend a Dropout layer before the given layer based on :cite:`galDropoutBayesian2016`."""
    if isinstance(rngs, int):
        rngs = Rngs(rngs)
    return Sequential(Dropout(p, rng_collection=rng_collection, rngs=rngs), obj)


register(Linear, prepend_flax_dropout)
