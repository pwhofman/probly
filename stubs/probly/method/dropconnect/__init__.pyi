"""DropConnect implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import dropconnect, dropconnect_traverser, register


## Torch
@dropconnect_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    ...


## Flax
@dropconnect_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    ...


__all__ = ["dropconnect", "dropconnect_traverser", "register"]
