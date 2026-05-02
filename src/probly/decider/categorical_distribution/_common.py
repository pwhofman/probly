"""Common deciders for categorical distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flextype import flexdispatch
from probly.representation.credal_set._common import CategoricalCredalSet
from probly.representation.distribution import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    create_categorical_distribution_from_logits,
)
from probly.representation.distribution._common import DirichletDistribution, GaussianDistribution

if TYPE_CHECKING:
    from probly.representation.representation import Representation


@flexdispatch
def categorical_from_mean(representation: Representation) -> CategoricalDistribution:
    """Create a categorical distribution from the mean of a representation."""
    msg = f"categorical_from_mean decider not supported for {type(representation).__name__}."
    raise NotImplementedError(msg)


@categorical_from_mean.register(CategoricalDistribution)
def _(distribution: CategoricalDistribution) -> CategoricalDistribution:
    return distribution


@categorical_from_mean.register(CategoricalDistributionSample)
def _(sample: CategoricalDistributionSample) -> CategoricalDistribution:
    return sample.sample_mean()


@categorical_from_mean.register(DirichletDistribution)
def _(distribution: DirichletDistribution) -> CategoricalDistribution:
    return distribution.mean


@categorical_from_mean.register(CategoricalCredalSet)
def _(credal_set: CategoricalCredalSet) -> CategoricalDistribution:
    return credal_set.barycenter


@categorical_from_mean.register(GaussianDistribution)
def _(gaussian: GaussianDistribution) -> CategoricalDistribution:
    return mean_field_categorical(gaussian, mean_field_factor=1.0)


@flexdispatch
def mean_field_categorical(representation: Representation, mean_field_factor: float = 1.0) -> CategoricalDistribution:
    """Mean-field approximation: representation -> CategoricalDistribution.

    For a Gaussian-over-logits, returns
    ``softmax(mean / sqrt(1 + mean_field_factor * var))``, the closed-form
    approximation to ``E_{m ~ N(mean, var)}[softmax(m)]``.

    Args:
        representation: The representation to convert.
        mean_field_factor: Scaling factor for the variance contribution to
            the logit denominator. Defaults to ``1.0``.

    Returns:
        A ``CategoricalDistribution`` approximating the predictive
        distribution.
    """
    msg = f"mean_field_categorical decider not supported for {type(representation).__name__}."
    raise NotImplementedError(msg)


@mean_field_categorical.register(GaussianDistribution)
def _(gaussian: GaussianDistribution, mean_field_factor: float = 1.0) -> CategoricalDistribution:
    # `**0.5` works on both torch.Tensor and np.ndarray. Backend-agnostic.
    scale = (1.0 + mean_field_factor * gaussian.var) ** 0.5
    return create_categorical_distribution_from_logits(gaussian.mean / scale)
