"""HetNets method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    HetNetPredictor,
    HetNetRepresentation,
    HetNetRepresenter,
    create_het_net_sample,
    het_net,
    het_net_traverser,
)


@het_net_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "HetNetPredictor",
    "HetNetRepresentation",
    "HetNetRepresenter",
    "create_het_net_sample",
    "het_net",
]
