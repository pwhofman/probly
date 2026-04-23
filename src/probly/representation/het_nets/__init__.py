"""HetNets representation."""

from __future__ import annotations

from probly.representation.sample.torch import TorchSample

from ._common import HetNetsRepresentation, create_het_nets_representation


@create_het_nets_representation.delayed_register(TorchSample)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "HetNetsRepresentation",
    "create_het_nets_representation",
]
