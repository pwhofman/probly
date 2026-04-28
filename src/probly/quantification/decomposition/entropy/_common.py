"""Base class for entropy-based decomposition methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import (
    AdditiveDecomposition,
    AleatoricDecomposition,
    CachingDecomposition,
)
from probly.quantification.measure.credal_set import lower_entropy, upper_entropy
from probly.quantification.measure.distribution import (
    conditional_entropy,
    entropy_of_expected_predictive_distribution,
    mutual_information,
)
from probly.representation.credal_set import CategoricalCredalSet
from probly.representation.distribution import Distribution, DistributionSample, SecondOrderDistribution
from probly.representation.het_nets._common import HetNetsRepresentation

if TYPE_CHECKING:
    from probly.quantification.measure.distribution import SecondOrderDistributionLike
    from probly.quantification.measure.distribution._common import LogBase


@decompose.register(SecondOrderDistribution | DistributionSample)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class SecondOrderEntropyDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Base class for entropy-based decomposition methods."""

    distribution: SecondOrderDistributionLike
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        """The total uncertainty of the decomposition."""
        return entropy_of_expected_predictive_distribution(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty of the decomposition."""
        return conditional_entropy(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty of the decomposition."""
        return mutual_information(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]


@decompose.register(CategoricalCredalSet)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class CredalSetEntropyDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Entropy decomposition for categorical credal sets.

    Total uncertainty is the upper entropy; aleatoric uncertainty is the lower
    entropy; epistemic uncertainty is their difference (upper minus lower).
    """

    credal_set: CategoricalCredalSet
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        return upper_entropy(self.credal_set, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        return lower_entropy(self.credal_set, base=self.base)  # ty:ignore[invalid-return-type]


@decompose.register(HetNetsRepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class LabelNoiseEntropyDecomposition[T](CachingDecomposition, AleatoricDecomposition[T]):
    """Entropy decomposition for HetNets representations.

    HetNets only capture aleatoric uncertainty.
    """

    distribution: SecondOrderDistributionLike
    base: LogBase = None

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty of the decomposition."""
        return conditional_entropy(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]


@decompose.register(Distribution)
@dataclass(frozen=True, slots=True)
class SingleDistributionEntropyDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Decomposition for a single distribution where total uncertainty is the entropy."""

    distribution: Distribution

    def __post_init__(self) -> None:
        object.__setattr__(self, "_caching", True)
        object.__setattr__(self, "_cache", {})

    @override
    @property
    def _total(self) -> T:
        """Total uncertainty is the entropy of the distribution."""
        return self.distribution.entropy()  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """Aleatoric uncertainty is the entropy for a single distribution."""
        return self._total
