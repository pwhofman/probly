"""Ordinal uncertainty decomposition methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.ordinal._common import (
    labelwise_conditional_entropy,
    labelwise_conditional_variance,
    labelwise_entropy_of_expected_predictive_distribution,
    labelwise_mutual_information_entropy,
    labelwise_mutual_information_variance,
    labelwise_variance_of_expected_predictive_distribution,
    ordinal_conditional_entropy,
    ordinal_conditional_variance,
    ordinal_entropy_of_expected_predictive_distribution,
    ordinal_mutual_information_entropy,
    ordinal_mutual_information_variance,
    ordinal_variance_of_expected_predictive_distribution,
)

if TYPE_CHECKING:
    from probly.quantification.measure.ordinal._common import LogBase, SecondOrderDistributionLike


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class OrdinalEntropyDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Entropy decomposition for ordinal classification models."""

    distribution: SecondOrderDistributionLike
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        """The total entropy uncertainty of the decomposition."""
        return ordinal_entropy_of_expected_predictive_distribution(self.distribution, base=self.base)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric entropy uncertainty of the decomposition."""
        return ordinal_conditional_entropy(self.distribution, base=self.base)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic entropy uncertainty of the decomposition."""
        return ordinal_mutual_information_entropy(self.distribution, base=self.base)  # ty: ignore[invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class OrdinalVarianceDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Variance decomposition for ordinal classification models."""

    distribution: SecondOrderDistributionLike

    @override
    @property
    def _total(self) -> T:
        """The total variance uncertainty of the decomposition."""
        return ordinal_variance_of_expected_predictive_distribution(self.distribution)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric variance uncertainty of the decomposition."""
        return ordinal_conditional_variance(self.distribution)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic variance uncertainty of the decomposition."""
        return ordinal_mutual_information_variance(self.distribution)  # ty: ignore[invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class LabelwiseBinaryEntropyDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Label-wise (one-vs-rest) binary entropy decomposition for classification models."""

    distribution: SecondOrderDistributionLike
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        """The total label-wise binary entropy uncertainty."""
        return labelwise_entropy_of_expected_predictive_distribution(self.distribution, base=self.base)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric label-wise binary entropy uncertainty."""
        return labelwise_conditional_entropy(self.distribution, base=self.base)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic label-wise binary entropy uncertainty."""
        return labelwise_mutual_information_entropy(self.distribution, base=self.base)  # ty: ignore[invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class LabelwiseBinaryVarianceDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Label-wise (one-vs-rest) binary variance decomposition for classification models."""

    distribution: SecondOrderDistributionLike

    @override
    @property
    def _total(self) -> T:
        """The total label-wise binary variance uncertainty."""
        return labelwise_variance_of_expected_predictive_distribution(self.distribution)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric label-wise binary variance uncertainty."""
        return labelwise_conditional_variance(self.distribution)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic label-wise binary variance uncertainty."""
        return labelwise_mutual_information_variance(self.distribution)  # ty: ignore[invalid-return-type]
