from __future__ import annotations

from typing import TYPE_CHECKING

from probly.representation.credal_set.torch import TorchConvexCredalSet, TorchProbabilityIntervalsCredalSet
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import compute_mean_probs

if TYPE_CHECKING:
    import torch


@compute_mean_probs.register(TorchSample)
def _(sample: TorchSample) -> torch.Tensor:
    """Compute the mean probabilities of a TorchSample."""
    return sample.sample_mean()


@compute_mean_probs.register(TorchCategoricalDistribution)
def _(distribution: TorchCategoricalDistribution) -> torch.Tensor:
    """Compute the mean probabilities of a TorchCategoricalDistribution."""
    return distribution.probabilities.mean(dim=-2)


@compute_mean_probs.register(TorchProbabilityIntervalsCredalSet)
def _(credal_set: TorchProbabilityIntervalsCredalSet) -> torch.Tensor:
    """Compute the mean probabilities of a TorchProbabilityIntervalsCredalSet."""
    return credal_set.lower_bounds + (credal_set.upper_bounds - credal_set.lower_bounds) / 2


@compute_mean_probs.register(TorchConvexCredalSet)
def _(credal_set: TorchConvexCredalSet) -> torch.Tensor:
    """Compute the mean probabilities of a TorchConvexCredalSet."""
    return credal_set.tensor.probabilities.mean(dim=-2)
