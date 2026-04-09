"""Representers for credal sets based on ensembles."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import CredalEnsemblingRepresenter, compute_representative_set


## Torch
@compute_representative_set.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "CredalEnsemblingRepresenter",
    "compute_representative_set",
]
