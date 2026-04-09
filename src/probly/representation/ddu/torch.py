"""Torch DDU representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution.torch_categorical import TorchTensorCategoricalDistribution

from ._common import DDURepresentation, create_ddu_representation

if TYPE_CHECKING:
    import torch


@create_ddu_representation.register(TorchTensorCategoricalDistribution)
@dataclass(frozen=True, slots=True)
class TorchDDURepresentation(DDURepresentation, TorchAxisProtected):
    """DDU representation backed by torch tensors.

    Args:
        softmax: Softmax probabilities over classes, shape (batch, num_classes).
        features: Feature vectors from the penultimate layer, shape (batch, feature_dim).
    """

    softmax: TorchTensorCategoricalDistribution
    densities: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"softmax": 0, "densities": 1}
