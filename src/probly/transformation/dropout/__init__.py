"""Dropout ensemble implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import DropoutPredictor, dropout, dropout_traverser, register


## Torch
@dropout_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@dropout_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = [
    "DropoutPredictor",
    "dropout",
    "dropout_traverser",
    "register",
]
