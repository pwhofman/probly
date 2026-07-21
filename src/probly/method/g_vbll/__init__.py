"""G-VBLL: Generative Variational Bayesian Last Layers implementation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    GVBLLPredictor,
    find_g_vbll_layer,
    g_vbll,
    g_vbll_traverser,
)


## Torch
@g_vbll_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "GVBLLPredictor",
    "find_g_vbll_layer",
    "g_vbll",
]
