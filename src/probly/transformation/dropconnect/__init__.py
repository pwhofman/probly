"""DropConnect implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import DropConnectPredictor, dropconnect, dropconnect_traverser, register


## Torch
@dropconnect_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@dropconnect_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = ["DropConnectPredictor", "dropconnect", "dropconnect_traverser", "register"]
