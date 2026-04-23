"""Torch HetNets representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.sample.torch import TorchSample

from ._common import HetNetsRepresentation, create_het_nets_representation

if TYPE_CHECKING:
    from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution


@create_het_nets_representation.register(TorchSample)
@dataclass(frozen=True, slots=True)
class HetNetsRepresentation(HetNetsRepresentation, TorchAxisProtected):
    """HetNets representation backed by torch tensors.

    Args:
        distribution: A sample of Softmax probabilities over classes.
    """

    distribution: TorchSample[TorchCategoricalDistribution]
    protected_axes: ClassVar[dict[str, int]] = {"distribution": 0}
