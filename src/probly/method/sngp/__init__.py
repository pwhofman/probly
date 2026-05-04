"""SNGP: Spectral-normalized Neural Gaussian Process implementation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    SNGPPredictor,
    SNGPRepresenter,
    compute_categorical_sample_from_logits,
    register,
    sngp,
    sngp_traverser,
)


## Torch
@sngp_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@compute_categorical_sample_from_logits.delayed_register((TORCH_TENSOR_LIKE, TORCH_TENSOR))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "SNGPPredictor",
    "SNGPRepresenter",
    "compute_categorical_sample_from_logits",
    "register",
    "sngp",
    "sngp_traverser",
]
