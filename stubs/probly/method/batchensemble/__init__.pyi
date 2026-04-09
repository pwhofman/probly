"""Batchensemble implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import batchensemble, batchensemble_traverser, register


## Torch
@batchensemble_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    ...


## Flax
@batchensemble_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    ...


__all__ = ["batchensemble", "batchensemble_traverser", "register"]
