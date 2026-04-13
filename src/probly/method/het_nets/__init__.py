"""HetNets implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import HetNetsPredictor, het_nets, hetnets_traverser, register


## Torch
@hetnets_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "HetNetsPredictor",
    "het_nets",
    "hetnets_traverser",
    "register",
]
