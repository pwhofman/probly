from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from flextype import flexdispatch

from probly.representation.distribution import CategoricalDistribution, CategoricalDistributionSample
from probly.utils import elementwise_entropy

if TYPE_CHECKING:
    from ndlike.array_like import WeakArrayLike

    from probly.lazy_types import NativeArray
    from probly.representation.distribution import Distribution, SecondOrderDistribution
    from probly.representation.sample import Sample


type LogBase = float | Literal["normalize"] | None

type SecondOrderDistributionLike = SecondOrderDistribution | Sample[Distribution]


@flexdispatch
def ordinal_variance(distribution: Distribution, base: LogBase = None) -> WeakArrayLike:
    """Compute the ordinal variance of a distribution."""
    msg = f"Ordinal variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_entropy(distribution: Distribution, base: LogBase = None) -> WeakArrayLike:
    """Compute the ordinal entropy of a distribution."""
    msg = f"Ordinal entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_variance_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the variance of the expected value of a second-order distribution."""
    msg = f"Variance of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_expected_conditional_variance(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the ordinal expected conditional variance of a distribution."""
    msg = f"Ordinal expected conditional variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_variance_of_conditional_mean(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the ordinal variance of the conditional mean of a distribution."""
    msg = f"Ordinal variance of conditional mean is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_entropy_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the entropy of the expected value of a second-order distribution."""
    msg = f"Entropy of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_mutual_information_entropy(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the mutual information of a distribution."""
    msg = f"Mutual information is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_conditional_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> WeakArrayLike:
    """Compute the conditional entropy of a distribution."""
    msg = f"Conditional entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def categorical_variance_total(distribution: SecondOrderDistributionLike, base: LogBase = None) -> WeakArrayLike:
    """Compute the total variance of an integer-encoded categorical sample."""
    msg = f"Ordinal integer variance total is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def categorical_variance_aleatoric(distribution: SecondOrderDistributionLike, base: LogBase = None) -> WeakArrayLike:
    """Compute the aleatoric variance of an integer-encoded categorical sample."""
    msg = f"Ordinal integer variance aleatoric is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_entropy(distribution: Distribution, base: LogBase = None) -> WeakArrayLike:
    """Compute the label-wise binary entropy of a distribution."""
    msg = f"Label-wise entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_variance(distribution: Distribution, base: LogBase = None) -> WeakArrayLike:
    """Compute the label-wise binary variance of a distribution."""
    msg = f"Label-wise variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_entropy_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the label-wise binary entropy of the expected predictive distribution."""
    msg = f"Label-wise entropy of EPD is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_conditional_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> WeakArrayLike:
    """Compute the label-wise conditional entropy of a distribution."""
    msg = f"Label-wise conditional entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_mutual_information_entropy(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the label-wise entropy-based mutual information of a distribution."""
    msg = f"Label-wise mutual information (entropy) is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_variance_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the label-wise binary variance of the expected predictive distribution."""
    msg = f"Label-wise variance of EPD is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_expected_conditional_variance(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the label-wise expected conditional variance of a distribution."""
    msg = f"Label-wise expected conditional variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_variance_of_conditional_mean(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the label-wise variance of the conditional mean of a distribution."""
    msg = f"Label-wise variance of conditional mean is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


# The variance measures reduce a categorical distribution to binary problems, the cumulative splits ``P(Y <= k)``
# for the ordinal measures and the one-vs-rest probabilities ``P(Y = k)`` for the label-wise ones, and sum the
# Bernoulli variances. They only need array operations that NumPy, PyTorch and JAX share, so one implementation
# on the backend-independent representation classes serves every backend.


def _cumulative_splits(probabilities: NativeArray) -> NativeArray:
    """Return the cumulative probabilities ``P(Y <= k)`` of shape ``(..., K - 1)``, leaving out the certain last one."""
    return probabilities.cumsum(-1)[..., :-1]


def _member_probabilities(sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]]) -> NativeArray:
    """Return the probabilities ``(M, ..., K)`` of the members of a sample, with the sample axis first."""
    return sample.samples.probabilities


def _binary_variance(binary_probabilities: NativeArray) -> NativeArray:
    """Sum the Bernoulli variances ``q (1 - q)`` of binary probabilities over the last axis."""
    q = binary_probabilities
    # Both operands come from q and therefore share its backend.
    variances: NativeArray = q * (1 - q)  # ty: ignore[unsupported-operator]
    return variances.sum(-1)


def _binary_variance_of_expected(member_binary_probabilities: NativeArray) -> NativeArray:
    """Compute the binary variance of the member mean, the total uncertainty."""
    return _binary_variance(member_binary_probabilities.mean(0))


def _expected_binary_variance(member_binary_probabilities: NativeArray) -> NativeArray:
    """Compute the mean binary variance of the members, the aleatoric uncertainty."""
    return _binary_variance(member_binary_probabilities).mean(0)


def _variance_of_binary_probabilities(member_binary_probabilities: NativeArray) -> NativeArray:
    """Compute the summed variance of the binary probabilities across members, the epistemic uncertainty.

    The population variance is spelled out because ``var`` defaults to the unbiased estimator in PyTorch and to the
    population variance in NumPy and JAX.
    """
    q = member_binary_probabilities
    # The member mean has the same backend as q and broadcasts over its first axis.
    centered: NativeArray = q - q.mean(0)  # ty: ignore[unsupported-operator]
    return (centered**2).mean(0).sum(-1)


@ordinal_variance.register(CategoricalDistribution)
def _categorical_ordinal_variance(
    distribution: CategoricalDistribution[NativeArray],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the ordinal variance of a categorical distribution."""
    return _binary_variance(_cumulative_splits(distribution.probabilities))


@ordinal_variance_of_expected_predictive_distribution.register(CategoricalDistributionSample)
def _categorical_sample_ordinal_variance_of_expected_predictive_distribution(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the ordinal variance of the expected value of a categorical sample."""
    return _binary_variance_of_expected(_cumulative_splits(_member_probabilities(sample)))


@ordinal_expected_conditional_variance.register(CategoricalDistributionSample)
def _categorical_sample_ordinal_expected_conditional_variance(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the ordinal expected conditional variance of a categorical sample."""
    return _expected_binary_variance(_cumulative_splits(_member_probabilities(sample)))


@ordinal_variance_of_conditional_mean.register(CategoricalDistributionSample)
def _categorical_sample_ordinal_variance_of_conditional_mean(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the ordinal variance of the conditional mean of a categorical sample."""
    return _variance_of_binary_probabilities(_cumulative_splits(_member_probabilities(sample)))


@labelwise_variance.register(CategoricalDistribution)
def _categorical_labelwise_variance(
    distribution: CategoricalDistribution[NativeArray],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the label-wise binary variance of a categorical distribution."""
    return _binary_variance(distribution.probabilities)


@labelwise_variance_of_expected_predictive_distribution.register(CategoricalDistributionSample)
def _categorical_sample_labelwise_variance_of_expected_predictive_distribution(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the label-wise binary variance of the expected predictive distribution."""
    return _binary_variance_of_expected(_member_probabilities(sample))


@labelwise_expected_conditional_variance.register(CategoricalDistributionSample)
def _categorical_sample_labelwise_expected_conditional_variance(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the label-wise expected conditional variance of a categorical sample."""
    return _expected_binary_variance(_member_probabilities(sample))


@labelwise_variance_of_conditional_mean.register(CategoricalDistributionSample)
def _categorical_sample_labelwise_variance_of_conditional_mean(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]],
    base: LogBase = None,  # noqa: ARG001
) -> NativeArray:
    """Compute the label-wise variance of the conditional mean of a categorical sample."""
    return _variance_of_binary_probabilities(_member_probabilities(sample))


def _binary_entropy(binary_probabilities: NativeArray, base: LogBase) -> NativeArray:
    """Sum the Bernoulli entropies of binary probabilities over the last axis.

    ``base="normalize"`` uses base 2, so that every binary problem contributes an entropy in ``[0, 1]``.
    """
    q = binary_probabilities
    # Clipping keeps a probability that exceeds one by rounding from contributing a negative entropy.
    entropies = (elementwise_entropy(q) + elementwise_entropy(1 - q)).clip(0.0) + 0.0  # ty: ignore[unsupported-operator]
    if base == "normalize":
        base = 2.0
    if base is not None and base != math.e:
        entropies = entropies / math.log(base)
    return entropies.sum(-1)


@ordinal_entropy.register(CategoricalDistribution)
def _categorical_ordinal_entropy(
    distribution: CategoricalDistribution[NativeArray], base: LogBase = None
) -> NativeArray:
    """Compute the ordinal entropy of a categorical distribution."""
    return _binary_entropy(_cumulative_splits(distribution.probabilities), base)


@ordinal_entropy_of_expected_predictive_distribution.register(CategoricalDistributionSample)
def _categorical_sample_ordinal_entropy_of_expected_predictive_distribution(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]], base: LogBase = None
) -> NativeArray:
    """Compute the ordinal entropy of the expected value of a categorical sample."""
    return _binary_entropy(_cumulative_splits(_member_probabilities(sample)).mean(0), base)


@ordinal_conditional_entropy.register(CategoricalDistributionSample)
def _categorical_sample_ordinal_conditional_entropy(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]], base: LogBase = None
) -> NativeArray:
    """Compute the ordinal conditional entropy of a categorical sample."""
    return _binary_entropy(_cumulative_splits(_member_probabilities(sample)), base).mean(0)


@ordinal_mutual_information_entropy.register(CategoricalDistributionSample)
def _categorical_sample_ordinal_mutual_information_entropy(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]], base: LogBase = None
) -> NativeArray:
    """Compute the ordinal mutual information (entropy-based) of a categorical sample."""
    total = _categorical_sample_ordinal_entropy_of_expected_predictive_distribution(sample, base)
    return total - _categorical_sample_ordinal_conditional_entropy(sample, base)


@labelwise_entropy.register(CategoricalDistribution)
def _categorical_labelwise_entropy(
    distribution: CategoricalDistribution[NativeArray], base: LogBase = None
) -> NativeArray:
    """Compute the label-wise binary entropy of a categorical distribution."""
    return _binary_entropy(distribution.probabilities, base)


@labelwise_entropy_of_expected_predictive_distribution.register(CategoricalDistributionSample)
def _categorical_sample_labelwise_entropy_of_expected_predictive_distribution(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]], base: LogBase = None
) -> NativeArray:
    """Compute the label-wise binary entropy of the expected predictive distribution."""
    return _binary_entropy(_member_probabilities(sample).mean(0), base)


@labelwise_conditional_entropy.register(CategoricalDistributionSample)
def _categorical_sample_labelwise_conditional_entropy(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]], base: LogBase = None
) -> NativeArray:
    """Compute the label-wise conditional entropy of a categorical sample."""
    return _binary_entropy(_member_probabilities(sample), base).mean(0)


@labelwise_mutual_information_entropy.register(CategoricalDistributionSample)
def _categorical_sample_labelwise_mutual_information_entropy(
    sample: CategoricalDistributionSample[CategoricalDistribution[NativeArray]], base: LogBase = None
) -> NativeArray:
    """Compute the label-wise entropy-based mutual information of a categorical sample."""
    total = _categorical_sample_labelwise_entropy_of_expected_predictive_distribution(sample, base)
    return total - _categorical_sample_labelwise_conditional_entropy(sample, base)
