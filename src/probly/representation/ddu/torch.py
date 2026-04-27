"""Torch DDU representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, override

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution

from ._common import DDURepresentation, create_ddu_representation

if TYPE_CHECKING:
    import torch


@create_ddu_representation.register(TorchCategoricalDistribution)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchDDURepresentation(DDURepresentation, TorchAxisProtected):
    """DDU representation backed by torch tensors.

    Args:
        softmax: Softmax probabilities over classes, shape (batch, num_classes).
        features: Feature vectors from the penultimate layer, shape (batch, feature_dim).
    """

    softmax: TorchCategoricalDistribution
    densities: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"softmax": 0, "densities": 1}

    @override
    @property
    def canonical_element(self) -> TorchCategoricalDistribution:
        """Return the canonical element of the DDU representation, which is the softmax distribution."""
        return self.softmax
