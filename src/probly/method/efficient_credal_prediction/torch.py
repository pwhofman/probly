"""Torch implementation of the efficient credal prediction method."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from torch import nn

from probly.predictor import RepresentationPredictor
from probly.representation.distribution import CategoricalDistribution

from ._common import efficient_credal_prediction_generator

if TYPE_CHECKING:
    import torch


@efficient_credal_prediction_generator.register(cls=nn.Module)
class TorchEfficientCredalPredictor[**In, Out: CategoricalDistribution](RepresentationPredictor[In, Out], Protocol):
    """A predictor that applies the efficient credal prediction method."""

    lower: torch.Tensor[float]
    upper: torch.Tensor[float]
