"""DARE: Deep Anti-Regularized Ensembles method API."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    DAREDecomposition,
    DarePredictor,
    DARERepresentation,
    DARERepresenter,
    dare,
    dare_anti_regularization,
)


## Torch
@dare_anti_regularization.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DAREDecomposition",
    "DARERepresentation",
    "DARERepresenter",
    "DarePredictor",
    "dare",
    "dare_anti_regularization",
]
