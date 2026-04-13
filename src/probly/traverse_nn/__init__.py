"""Traverser utilities for neural networks."""

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import LAYER_COUNT, compose as nn_compose, is_first_layer, layer_count_traverser, nn_traverser
from .reset_traverser import reset_traverser


## Torch
@nn_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@nn_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = [
    "LAYER_COUNT",
    "is_first_layer",
    "layer_count_traverser",
    "nn_compose",
    "nn_traverser",
    "reset_traverser",
]
