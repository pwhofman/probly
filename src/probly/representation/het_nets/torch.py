"""Torch HetNets representation."""

from __future__ import annotations

from dataclasses import dataclass

from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)

from ._common import HetNetsRepresentation, create_het_nets_representation


@dataclass(frozen=True, slots=True)
class TorchHetNetsRepresentation(  # ty:ignore[conflicting-metaclass]
    HetNetsRepresentation[TorchCategoricalDistribution],
    TorchCategoricalDistributionSample,
):
    """HetNets representation backed by torch tensors."""


create_het_nets_representation.register(
    TorchCategoricalDistribution,
    TorchHetNetsRepresentation.from_iterable,
)
