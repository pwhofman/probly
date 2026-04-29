"""SNGP representer."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import SNGPRepresenter, compute_categorical_sample_from_logits


## Torch
@compute_categorical_sample_from_logits.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["SNGPRepresenter", "compute_categorical_sample_from_logits"]
