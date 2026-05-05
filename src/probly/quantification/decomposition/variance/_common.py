"""Base class for variance-based decomposition methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.ordinal._common import (
    categorical_variance_aleatoric,
    categorical_variance_total,
)
from probly.quantification.measure.variance._common import (
    conditional_variance,
    mutual_information_variance,
    variance_of_expected_predictive_distribution,
)

if TYPE_CHECKING:
    from probly.quantification.measure.distribution import SecondOrderDistributionLike
    from probly.quantification.measure.distribution._common import LogBase


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class SecondOrderVarianceDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Variance decomposition for regression models in an ensemble or sampling setting.

    Follows the Law of Total Variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X]).
    """

    distribution: SecondOrderDistributionLike
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        """The total variance (overall predictive uncertainty)."""
        return variance_of_expected_predictive_distribution(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric variance (expected variance over the ensemble/samples)."""
        return conditional_variance(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic variance (variance of the means over the ensemble/samples)."""
        return mutual_information_variance(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class CategoricalVarianceDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Variance decomposition for integer-encoded categorical samples."""

    distribution: SecondOrderDistributionLike
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        """The total variance (overall predictive uncertainty)."""
        return categorical_variance_total(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric variance (expected variance over the ensemble/samples)."""
        return categorical_variance_aleatoric(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]

