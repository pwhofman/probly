"""Representers for credal sets based on ensembles."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    CredalBNNRepresenter,
    CredalEnsemblingRepresenter,
    CredalRelativeLikelihoodRepresenter,
    CredalWrapperRepresenter,
    compute_credal_ensembling_set,
    compute_credal_net_set,
)


## Torch
@compute_credal_ensembling_set.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@compute_credal_net_set.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "CredalBNNRepresenter",
    "CredalEnsemblingRepresenter",
    "CredalRelativeLikelihoodRepresenter",
    "CredalWrapperRepresenter",
    "compute_credal_ensembling_set",
    "compute_credal_net_set",
]
