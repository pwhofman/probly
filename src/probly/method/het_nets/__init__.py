"""HetNets method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    HetNetsPredictor,
    HetNetsRepresentation,
    HetNetsRepresenter,
    create_het_nets_sample,
    het_nets,
    het_nets_traverser,
)


@het_nets_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "HetNetsPredictor",
    "HetNetsRepresentation",
    "HetNetsRepresenter",
    "create_het_nets_sample",
    "het_nets",
]
