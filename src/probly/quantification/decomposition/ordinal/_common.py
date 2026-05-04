"""Ordinal uncertainty decomposition methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.ordinal._common import (
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
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        """The total variance uncertainty of the decomposition."""
        return ordinal_variance_of_expected_predictive_distribution(self.distribution, base=self.base)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric variance uncertainty of the decomposition."""
        return ordinal_conditional_variance(self.distribution, base=self.base)  # ty: ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic variance uncertainty of the decomposition."""
        return ordinal_mutual_information_variance(self.distribution, base=self.base)  # ty: ignore[invalid-return-type]
