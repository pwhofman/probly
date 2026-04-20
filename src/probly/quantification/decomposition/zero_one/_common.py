"""Base class for zero-one proper scoring rule decomposition methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.distribution._common import (
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
)

if TYPE_CHECKING:
    from probly.quantification.measure.distribution._common import SecondOrderDistributionLike


@dataclass(frozen=True, slots=True)
class SecondOrderZeroOneDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Zero-one proper scoring rule decomposition of second-order uncertainty.

    The components are defined for a second-order categorical distribution Q with Bayesian
    model average ``theta_bar = E_{theta ~ Q}[theta]``:

    - Total uncertainty: ``1 - max_k theta_bar_k``.
    - Aleatoric uncertainty: ``E_{theta ~ Q}[1 - max_k theta_k]``.
    - Epistemic uncertainty: ``E_{theta ~ Q}[max_k theta_k - theta_{argmax_k theta_bar_k}]``.

    The decomposition is additive: ``total == aleatoric + epistemic``.
    """

    distribution: SecondOrderDistributionLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "_caching", True)
        object.__setattr__(self, "_cache", {})

    @override
    @property
    def _total(self) -> T:
        """The total uncertainty of the decomposition."""
        return max_probability_complement_of_expected(self.distribution)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty of the decomposition."""
        return expected_max_probability_complement(self.distribution)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty of the decomposition."""
        return max_disagreement(self.distribution)  # ty:ignore[invalid-return-type]
