"""Traverser utilities for neural networks."""

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from . import _common

## NN

LAYER_COUNT = _common.LAYER_COUNT
is_first_layer = _common.is_first_layer

layer_count_traverser = _common.layer_count_traverser
nn_traverser = _common.nn_traverser

nn_compose = _common.compose


## Torch
@nn_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@nn_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415
