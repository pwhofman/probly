"""Ordinal classification and regression uncertainty decompositions.

Implements the binary-reduction decompositions of :cite:`haasOrdinal2025`:

* Order-Consistent Split (OCS) reduction with binary entropy or variance.
* Label-wise (one-vs-rest) binary reduction with entropy or variance.
* Standard categorical variance via the law of total variance.
* Gaussian regression variance via the law of total variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from flextype import flexdispatch
from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.representation.distribution import CategoricalDistributionSample, GaussianDistributionSample

if TYPE_CHECKING:
    from probly.quantification.measure.distribution._common import LogBase
    from probly.representation.array_like import ArrayLike


# OCS (Order-Consistent Split) binary reduction


@flexdispatch
def ordinal_binary_entropy_total(
    sample: CategoricalDistributionSample,
    base: LogBase = None,
) -> ArrayLike:
    """Compute the OCS binary-entropy total uncertainty of a categorical sample.

    For ``K`` ordinal classes the cumulative probabilities ``p_<=k`` for
    ``k = 1, ..., K-1`` define ``K-1`` binary problems. The total uncertainty is
    the sum of binary entropies of the ensemble-averaged cumulative probabilities.

    Args:
        sample: Sample from a second-order categorical distribution.
        base: Logarithm base for the binary entropy. ``None`` (the default) uses
            natural logarithms; a float specifies the base directly; ``"normalize"``
            divides by ``log(2)`` so that each binary entropy lies in ``[0, 1]``.

    Returns:
        Array of total uncertainties with the sample axis reduced.
    """
    msg = f"Ordinal binary entropy total is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_binary_entropy_aleatoric(
    sample: CategoricalDistributionSample,
    base: LogBase = None,
) -> ArrayLike:
    """Compute the OCS binary-entropy aleatoric uncertainty of a categorical sample.

    The aleatoric component is the ensemble mean of the per-model sum of binary
    entropies of the cumulative probabilities ``p_<=k``.

    Args:
        sample: Sample from a second-order categorical distribution.
        base: Logarithm base for the binary entropy (see
            :func:`ordinal_binary_entropy_total`).

    Returns:
        Array of aleatoric uncertainties with the sample axis reduced.
    """
    msg = f"Ordinal binary entropy aleatoric is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_binary_variance_total(sample: CategoricalDistributionSample) -> ArrayLike:
    """Compute the OCS binary-variance total uncertainty of a categorical sample.

    For each cumulative probability ``p_<=k`` the binary variance of the
    ensemble-averaged probability is summed across ``k = 1, ..., K-1``.

    Args:
        sample: Sample from a second-order categorical distribution.

    Returns:
        Array of total uncertainties with the sample axis reduced.
    """
    msg = f"Ordinal binary variance total is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_binary_variance_aleatoric(sample: CategoricalDistributionSample) -> ArrayLike:
    """Compute the OCS binary-variance aleatoric uncertainty of a categorical sample.

    The aleatoric component is the ensemble mean of the per-model sum of binary
    variances ``p_<=k * (1 - p_<=k)``.

    Args:
        sample: Sample from a second-order categorical distribution.

    Returns:
        Array of aleatoric uncertainties with the sample axis reduced.
    """
    msg = f"Ordinal binary variance aleatoric is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


# Label-wise (one-vs-rest) binary reduction


@flexdispatch
def labelwise_binary_entropy_total(
    sample: CategoricalDistributionSample,
    base: LogBase = None,
) -> ArrayLike:
    """Compute the label-wise binary-entropy total uncertainty of a categorical sample.

    For ``K`` classes each label defines a one-vs-rest binary problem with
    probability ``p_k``. The total uncertainty is the sum of binary entropies of
    the ensemble-averaged ``p_k``.

    Args:
        sample: Sample from a second-order categorical distribution.
        base: Logarithm base for the binary entropy (see
            :func:`ordinal_binary_entropy_total`).

    Returns:
        Array of total uncertainties with the sample axis reduced.
    """
    msg = f"Labelwise binary entropy total is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_binary_entropy_aleatoric(
    sample: CategoricalDistributionSample,
    base: LogBase = None,
) -> ArrayLike:
    """Compute the label-wise binary-entropy aleatoric uncertainty of a categorical sample.

    The aleatoric component is the ensemble mean of the per-model sum of binary
    entropies ``H(p_k)``.

    Args:
        sample: Sample from a second-order categorical distribution.
        base: Logarithm base for the binary entropy (see
            :func:`ordinal_binary_entropy_total`).

    Returns:
        Array of aleatoric uncertainties with the sample axis reduced.
    """
    msg = f"Labelwise binary entropy aleatoric is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_binary_variance_total(sample: CategoricalDistributionSample) -> ArrayLike:
    """Compute the label-wise binary-variance total uncertainty of a categorical sample.

    The total uncertainty is the sum across labels of the binary variance of the
    ensemble-averaged probability ``p_k``.

    Args:
        sample: Sample from a second-order categorical distribution.

    Returns:
        Array of total uncertainties with the sample axis reduced.
    """
    msg = f"Labelwise binary variance total is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_binary_variance_aleatoric(sample: CategoricalDistributionSample) -> ArrayLike:
    """Compute the label-wise binary-variance aleatoric uncertainty of a categorical sample.

    The aleatoric component is the ensemble mean of the per-model sum of binary
    variances ``p_k * (1 - p_k)``.

    Args:
        sample: Sample from a second-order categorical distribution.

    Returns:
        Array of aleatoric uncertainties with the sample axis reduced.
    """
    msg = f"Labelwise binary variance aleatoric is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


# Standard categorical variance via law of total variance


@flexdispatch
def categorical_variance_total(sample: CategoricalDistributionSample) -> ArrayLike:
    """Compute the total variance of an integer-encoded categorical sample.

    Encodes labels as ``1, ..., K`` and applies the law of total variance to the
    Bayesian model average. Concretely, the result is
    ``sum_k (k - mu)^2 * p_bar_k`` where ``p_bar = mean_m[p_m]`` and
    ``mu = sum_k k * p_bar_k``.

    Args:
        sample: Sample from a second-order categorical distribution.

    Returns:
        Array of total variances with the sample axis reduced.
    """
    msg = f"Categorical variance total is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


@flexdispatch
def categorical_variance_aleatoric(sample: CategoricalDistributionSample) -> ArrayLike:
    """Compute the aleatoric variance of an integer-encoded categorical sample.

    Returns ``mean_m[ sum_k (k - mu_m)^2 * p_k^m ]``, the ensemble mean of the
    per-model variances under the integer label encoding ``1, ..., K``.

    Args:
        sample: Sample from a second-order categorical distribution.

    Returns:
        Array of aleatoric variances with the sample axis reduced.
    """
    msg = f"Categorical variance aleatoric is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


# Gaussian regression variance via law of total variance


@flexdispatch
def gaussian_variance_aleatoric(sample: GaussianDistributionSample) -> ArrayLike:
    """Compute the aleatoric variance of a Gaussian regression sample.

    The aleatoric component is the ensemble mean of the predictive variances
    ``mean_m[sigma_m^2]``.

    Args:
        sample: Sample from a second-order Gaussian distribution.

    Returns:
        Array of aleatoric variances with the sample axis reduced.
    """
    msg = f"Gaussian variance aleatoric is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


@flexdispatch
def gaussian_variance_epistemic(sample: GaussianDistributionSample) -> ArrayLike:
    """Compute the epistemic variance of a Gaussian regression sample.

    The epistemic component is the variance of the predicted means ``var_m[mu_m]``.

    Args:
        sample: Sample from a second-order Gaussian distribution.

    Returns:
        Array of epistemic variances with the sample axis reduced.
    """
    msg = f"Gaussian variance epistemic is not supported for samples of type {type(sample)}."
    raise NotImplementedError(msg)


# Decomposition classes


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class OrdinalEntropyDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Order-Consistent Split (OCS) entropy decomposition for ordinal classification.

    Reduces a ``K``-class ordinal problem to ``K-1`` binary problems via the
    cumulative probabilities ``p_<=k`` and uses binary Shannon entropy as the base
    measure :cite:`haasOrdinal2025`. The decomposition is additive:
    ``total = aleatoric + epistemic``.
    """

    sample: CategoricalDistributionSample
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        return ordinal_binary_entropy_total(self.sample, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        return ordinal_binary_entropy_aleatoric(self.sample, base=self.base)  # ty:ignore[invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class OrdinalVarianceDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Order-Consistent Split (OCS) variance decomposition for ordinal classification.

    Reduces a ``K``-class ordinal problem to ``K-1`` binary problems via the
    cumulative probabilities ``p_<=k`` and uses binary variance ``p(1-p)`` as the
    base measure :cite:`haasOrdinal2025`. The decomposition is additive:
    ``total = aleatoric + epistemic``.
    """

    sample: CategoricalDistributionSample

    @override
    @property
    def _total(self) -> T:
        return ordinal_binary_variance_total(self.sample)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        return ordinal_binary_variance_aleatoric(self.sample)  # ty:ignore[invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class LabelwiseBinaryEntropyDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Label-wise (one-vs-rest) binary-entropy decomposition for classification.

    Reduces a ``K``-class problem to ``K`` independent one-vs-rest binary problems
    and uses binary Shannon entropy as the base measure :cite:`haasOrdinal2025`.
    The decomposition is additive: ``total = aleatoric + epistemic``.
    """

    sample: CategoricalDistributionSample
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        return labelwise_binary_entropy_total(self.sample, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        return labelwise_binary_entropy_aleatoric(self.sample, base=self.base)  # ty:ignore[invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class LabelwiseBinaryVarianceDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Label-wise (one-vs-rest) binary-variance decomposition for classification.

    Reduces a ``K``-class problem to ``K`` independent one-vs-rest binary problems
    and uses binary variance ``p(1-p)`` as the base measure
    :cite:`haasOrdinal2025`. The decomposition is additive:
    ``total = aleatoric + epistemic``.
    """

    sample: CategoricalDistributionSample

    @override
    @property
    def _total(self) -> T:
        return labelwise_binary_variance_total(self.sample)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        return labelwise_binary_variance_aleatoric(self.sample)  # ty:ignore[invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class CategoricalVarianceDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Variance decomposition for integer-encoded categorical (ordinal) samples.

    Encodes labels as ``1, ..., K`` and applies the law of total variance to the
    second-order distribution: ``total = aleatoric + epistemic`` where the
    aleatoric component is the expected per-model variance and the epistemic
    component is the variance of the per-model means :cite:`haasOrdinal2025`.
    """

    sample: CategoricalDistributionSample

    @override
    @property
    def _total(self) -> T:
        return categorical_variance_total(self.sample)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        return categorical_variance_aleatoric(self.sample)  # ty:ignore[invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class GaussianVarianceDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Variance decomposition for Gaussian regression samples.

    Applies the law of total variance to a sample of per-model Gaussian
    predictions: the aleatoric component is the mean predictive variance
    ``mean_m[sigma_m^2]`` and the epistemic component is the variance of the
    predictive means ``var_m[mu_m]``. The total variance is computed as the sum
    of the two, mirroring the law of total variance.
    """

    sample: GaussianDistributionSample

    @override
    @property
    def _aleatoric(self) -> T:
        return gaussian_variance_aleatoric(self.sample)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        return gaussian_variance_epistemic(self.sample)  # ty:ignore[invalid-return-type]
