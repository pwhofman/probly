"""Torch dropout implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax.nnx import Dropout, Linear, Sequential

from .common import register

if TYPE_CHECKING:
    from collections.abc import Callable


def prepend_flax_dropout(obj: Callable, p: float) -> Sequential:
    """Prepend a Dropout layer before the given layer based on :cite:`galDropoutBayesian2016`."""
    return Sequential(Dropout(p), obj)


register(Linear, prepend_flax_dropout)
