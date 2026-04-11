"""Base class for entropy-based decomposition methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification._quantification import quantify
from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.distribution._common import (
    conditional_entropy,
    entropy_of_expected_value,
    mutual_information,
)
from probly.representation.distribution._common import DistributionSample, SecondOrderDistribution

if TYPE_CHECKING:
    from probly.quantification.measure.distribution._common import SecondOrderDistributionLike


@quantify.register(SecondOrderDistribution | DistributionSample)
@dataclass(frozen=True, slots=True)
class SecondOrderEntropyDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Base class for entropy-based decomposition methods."""

    distribution: SecondOrderDistributionLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "_caching", True)
        object.__setattr__(self, "_cache", {})

    @override
    @property
    def _total(self) -> T:
        """The total uncertainty of the decomposition."""
        return entropy_of_expected_value(self.distribution)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty of the decomposition."""
        return conditional_entropy(self.distribution)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty of the decomposition."""
        return mutual_information(self.distribution)  # ty:ignore[invalid-return-type]
