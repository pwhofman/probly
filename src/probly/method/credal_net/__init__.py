"""Credal net implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import CredalNetPredictor, credal_net, credal_net_traverser, intersection_probability


## Torch
@credal_net_traverser.delayed_register(TORCH_MODULE)
@intersection_probability.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "CredalNetPredictor",
    "credal_net",
    "intersection_probability",
]
