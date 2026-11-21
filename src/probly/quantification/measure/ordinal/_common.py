from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from flextype import flexdispatch

from probly.representation.distribution import CategoricalDistribution, CategoricalDistributionSample

if TYPE_CHECKING:
    from ndlike.array_like import ArrayLike

    from probly.representation.distribution import Distribution, SecondOrderDistribution
    from probly.representation.sample import Sample


type LogBase = float | Literal["normalize"] | None

type SecondOrderDistributionLike = SecondOrderDistribution | Sample[Distribution]


@flexdispatch
def ordinal_variance(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the ordinal variance of a distribution."""
    msg = f"Ordinal variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_entropy(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the ordinal entropy of a distribution."""
    msg = f"Ordinal entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_variance_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the variance of the expected value of a second-order distribution."""
    msg = f"Variance of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_expected_conditional_variance(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the ordinal expected conditional variance of a distribution."""
    msg = f"Ordinal expected conditional variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_variance_of_conditional_mean(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the ordinal variance of the conditional mean of a distribution."""
    msg = f"Ordinal variance of conditional mean is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_entropy_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the entropy of the expected value of a second-order distribution."""
    msg = f"Entropy of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_mutual_information_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the mutual information of a distribution."""
    msg = f"Mutual information is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_conditional_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the conditional entropy of a distribution."""
    msg = f"Conditional entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def categorical_variance_total(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the total variance of an integer-encoded categorical sample."""
    msg = f"Ordinal integer variance total is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def categorical_variance_aleatoric(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the aleatoric variance of an integer-encoded categorical sample."""
    msg = f"Ordinal integer variance aleatoric is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_entropy(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise binary entropy of a distribution."""
    msg = f"Label-wise entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_variance(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise binary variance of a distribution."""
    msg = f"Label-wise variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_entropy_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the label-wise binary entropy of the expected predictive distribution."""
    msg = f"Label-wise entropy of EPD is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_conditional_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise conditional entropy of a distribution."""
    msg = f"Label-wise conditional entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_mutual_information_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise entropy-based mutual information of a distribution."""
    msg = f"Label-wise mutual information (entropy) is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_variance_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the label-wise binary variance of the expected predictive distribution."""
    msg = f"Label-wise variance of EPD is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_expected_conditional_variance(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the label-wise expected conditional variance of a distribution."""
    msg = f"Label-wise expected conditional variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_variance_of_conditional_mean(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the label-wise variance of the conditional mean of a distribution."""
    msg = f"Label-wise variance of conditional mean is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


# The variance measures reduce a categorical distribution to binary problems, the cumulative splits ``P(Y <= k)``
# for the ordinal measures and the one-vs-rest probabilities ``P(Y = k)`` for the label-wise ones, and sum the
# Bernoulli variances. They only need array operations that NumPy, PyTorch and JAX share, so one implementation
# on the backend-independent representation classes serves every backend.


def _cumulative_splits(probabilities: object) -> Any:  # noqa: ANN401
    """Return the cumulative probabilities ``P(Y <= k)`` of shape ``(..., K - 1)``, leaving out the certain last one."""
    return cast("Any", probabilities).cumsum(axis=-1)[..., :-1]


def _member_probabilities(sample: CategoricalDistributionSample) -> Any:  # noqa: ANN401
    """Return the probabilities ``(M, ..., K)`` of the members of a sample, with the sample axis first."""
    return cast("Any", sample.samples).probabilities


def _binary_variance(binary_probabilities: object) -> ArrayLike:
    """Sum the Bernoulli variances ``q (1 - q)`` of binary probabilities over the last axis."""
    q = cast("Any", binary_probabilities)
    return (q * (1 - q)).sum(axis=-1)


def _binary_variance_of_expected(member_binary_probabilities: object) -> ArrayLike:
    """Compute the binary variance of the member mean, the total uncertainty."""
    return _binary_variance(cast("Any", member_binary_probabilities).mean(axis=0))


def _expected_binary_variance(member_binary_probabilities: object) -> ArrayLike:
    """Compute the mean binary variance of the members, the aleatoric uncertainty."""
    return cast("Any", _binary_variance(member_binary_probabilities)).mean(axis=0)


def _variance_of_binary_probabilities(member_binary_probabilities: object) -> ArrayLike:
    """Compute the summed variance of the binary probabilities across members, the epistemic uncertainty.

    The population variance is spelled out because ``var`` defaults to the unbiased estimator in PyTorch and to the
    population variance in NumPy and JAX.
    """
    q = cast("Any", member_binary_probabilities)
    return ((q - q.mean(axis=0, keepdims=True)) ** 2).mean(axis=0).sum(axis=-1)


@ordinal_variance.register(CategoricalDistribution)
def _categorical_ordinal_variance(
    distribution: CategoricalDistribution,
    base: LogBase = None,  # noqa: ARG001
) -> ArrayLike:
    """Compute the ordinal variance of a categorical distribution."""
    return _binary_variance(_cumulative_splits(distribution.probabilities))


@ordinal_variance_of_expected_predictive_distribution.register(CategoricalDistributionSample)
def _categorical_sample_ordinal_variance_of_expected_predictive_distribution(
    sample: CategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> ArrayLike:
    """Compute the ordinal variance of the expected value of a categorical sample."""
    return _binary_variance_of_expected(_cumulative_splits(_member_probabilities(sample)))


@ordinal_expected_conditional_variance.register(CategoricalDistributionSample)
def _categorical_sample_ordinal_expected_conditional_variance(
    sample: CategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> ArrayLike:
    """Compute the ordinal expected conditional variance of a categorical sample."""
    return _expected_binary_variance(_cumulative_splits(_member_probabilities(sample)))


@ordinal_variance_of_conditional_mean.register(CategoricalDistributionSample)
def _categorical_sample_ordinal_variance_of_conditional_mean(
    sample: CategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> ArrayLike:
    """Compute the ordinal variance of the conditional mean of a categorical sample."""
    return _variance_of_binary_probabilities(_cumulative_splits(_member_probabilities(sample)))


@labelwise_variance.register(CategoricalDistribution)
def _categorical_labelwise_variance(
    distribution: CategoricalDistribution,
    base: LogBase = None,  # noqa: ARG001
) -> ArrayLike:
    """Compute the label-wise binary variance of a categorical distribution."""
    return _binary_variance(distribution.probabilities)


@labelwise_variance_of_expected_predictive_distribution.register(CategoricalDistributionSample)
def _categorical_sample_labelwise_variance_of_expected_predictive_distribution(
    sample: CategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> ArrayLike:
    """Compute the label-wise binary variance of the expected predictive distribution."""
    return _binary_variance_of_expected(_member_probabilities(sample))


@labelwise_expected_conditional_variance.register(CategoricalDistributionSample)
def _categorical_sample_labelwise_expected_conditional_variance(
    sample: CategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> ArrayLike:
    """Compute the label-wise expected conditional variance of a categorical sample."""
    return _expected_binary_variance(_member_probabilities(sample))


@labelwise_variance_of_conditional_mean.register(CategoricalDistributionSample)
def _categorical_sample_labelwise_variance_of_conditional_mean(
    sample: CategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> ArrayLike:
    """Compute the label-wise variance of the conditional mean of a categorical sample."""
    return _variance_of_binary_probabilities(_member_probabilities(sample))
