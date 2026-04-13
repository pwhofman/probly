"""HetNets implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import hetnets_traverser, HetNetsPredictor, het_nets, register

## Torch
@hetnets_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415

__all__ = [
    "hetnets_traverser",
    "HetNetsPredictor",
    "het_nets",
    "register",
]

