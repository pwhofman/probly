"""Decomposition of second-order uncertainty for an arbitrary proper scoring rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.distribution._common import (
    expected_generalized_entropy,
    generalized_entropy_of_expected,
)

if TYPE_CHECKING:
    from probly.quantification.measure.distribution._common import SecondOrderDistributionLike
    from probly.quantification.scoring_rule import ProperScoringRule


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class SecondOrderScoringRuleDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Total/aleatoric/epistemic decomposition induced by a proper scoring rule.

    For a second-order categorical distribution ``Q`` with Bayesian model average
    ``theta_bar = E[theta]`` and generalized entropy ``G(theta) = <theta, loss(theta)>``:

    - Total uncertainty: ``G(theta_bar)``.
    - Aleatoric uncertainty: ``E_{theta ~ Q}[G(theta)]``.
    - Epistemic uncertainty: ``total - aleatoric`` (a non-negative Jensen gap).

    The decomposition is additive: ``total == aleatoric + epistemic``.
    """

    distribution: SecondOrderDistributionLike
    scoring_rule: ProperScoringRule

    @override
    @property
    def _total(self) -> T:
        """The total uncertainty of the decomposition."""
        return generalized_entropy_of_expected(self.distribution, self.scoring_rule)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty of the decomposition."""
        return expected_generalized_entropy(self.distribution, self.scoring_rule)  # ty:ignore[invalid-return-type]
