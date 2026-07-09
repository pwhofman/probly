"""VBLL: Variational Bayesian Last Layers implementation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    VBLLPredictor,
    VBLLRepresenter,
    compute_vbll_categorical_sample,
    find_vbll_layer,
    vbll,
    vbll_traverser,
)


## Torch
@vbll_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@compute_vbll_categorical_sample.delayed_register((TORCH_TENSOR_LIKE, TORCH_TENSOR))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "VBLLPredictor",
    "VBLLRepresenter",
    "find_vbll_layer",
    "vbll",
]
