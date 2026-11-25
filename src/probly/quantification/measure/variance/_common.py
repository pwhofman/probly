from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from flextype import flexdispatch

from probly.representation.distribution import Distribution, GaussianDistribution, GaussianDistributionSample
from probly.representation.sample import Sample

if TYPE_CHECKING:
    from ndlike.array_like import WeakArrayLike

    from probly.lazy_types import NativeArray
    from probly.representation.distribution import SecondOrderDistribution


type LogBase = float | Literal["normalize"] | None


@flexdispatch
def variance(distribution: Distribution, base: LogBase = None) -> WeakArrayLike:
    """Compute the variance of a distribution."""
    msg = f"Variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


type SecondOrderDistributionLike = SecondOrderDistribution | Sample[Distribution]


@flexdispatch
def variance_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the variance of the expected value of a second-order distribution."""
    msg = f"Variance of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def expected_conditional_variance(distribution: SecondOrderDistributionLike, base: LogBase = None) -> WeakArrayLike:
    """Compute the expected conditional variance of a second-order distribution."""
    msg = f"Expected conditional variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def variance_of_conditional_mean(distribution: SecondOrderDistributionLike, base: LogBase = None) -> WeakArrayLike:
    """Compute the variance of the conditional mean of a second-order distribution."""
    msg = f"Variance of conditional mean is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


def _variance_over_members(members: NativeArray) -> NativeArray:
    """Compute the population variance over the leading sample axis.

    Spelled out because ``var`` defaults to the unbiased estimator in PyTorch and to the population variance in
    NumPy and JAX.
    """
    centered: NativeArray = members - members.mean(0)  # ty: ignore[unsupported-operator]
    return (centered**2).mean(0)


@variance.register(GaussianDistribution)
def _gaussian_variance[A: NativeArray](
    distribution: GaussianDistribution[A],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Return the variance of a Gaussian distribution."""
    return distribution.var


@expected_conditional_variance.register(GaussianDistributionSample)
def _gaussian_sample_expected_conditional_variance[A: NativeArray](
    sample: GaussianDistributionSample[GaussianDistribution[A]],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the aleatoric variance of a Gaussian sample, the mean of the member variances."""
    return sample.samples.var.mean(0)


@variance_of_conditional_mean.register(GaussianDistributionSample)
def _gaussian_sample_variance_of_conditional_mean[A: NativeArray](
    sample: GaussianDistributionSample[GaussianDistribution[A]],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the epistemic variance of a Gaussian sample, the variance of the member means."""
    return _variance_over_members(sample.samples.mean)


@variance_of_expected_predictive_distribution.register(GaussianDistributionSample)
def _gaussian_sample_variance_of_expected_predictive_distribution[A: NativeArray](
    sample: GaussianDistributionSample[GaussianDistribution[A]],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the total predictive variance of a Gaussian sample via the law of total variance."""
    aleatoric = _gaussian_sample_expected_conditional_variance(sample)
    epistemic = _gaussian_sample_variance_of_conditional_mean(sample)
    return aleatoric + epistemic


@variance_of_expected_predictive_distribution.register(Sample)
@variance_of_conditional_mean.register(Sample)
def _sample_variance(
    sample: Sample[NativeArray, NativeArray],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the variance of a sample of point predictions, which is all epistemic."""
    return _variance_over_members(sample.samples)
