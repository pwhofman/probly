"""HetNets representation."""

from __future__ import annotations

from probly.lazy_types import TORCH_SAMPLE

from ._common import HetNetsRepresentation, create_het_nets_representation


@create_het_nets_representation.delayed_register(TORCH_SAMPLE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "HetNetsRepresentation",
    "create_het_nets_representation",
]
