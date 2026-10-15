"""Decomposition of second-order uncertainty for an arbitrary scoring rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.distribution._common import (
    expected_generalized_entropy,
    generalized_entropy_of_expected,
)

if TYPE_CHECKING:
    from probly.metrics import ScoringRule
    from probly.quantification.measure.distribution._common import SecondOrderDistributionLike


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class SecondOrderScoringRuleDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Total/aleatoric/epistemic decomposition induced by a per-instance scoring rule.

    The scoring rule is a function such as :func:`probly.metrics.log_loss_per_instance`. For a second-order
    categorical distribution ``Q`` with Bayesian model average ``theta_bar = E[theta]`` and generalized entropy
    ``G(theta) = scoring_rule(theta, theta)``, the expected loss of ``theta`` under itself:

    - Total uncertainty: ``G(theta_bar)``.
    - Aleatoric uncertainty: ``E_{theta ~ Q}[G(theta)]``.
    - Epistemic uncertainty: ``total - aleatoric`` (a Jensen gap, non-negative when the rule is proper).

    The decomposition is additive: ``total == aleatoric + epistemic``.

    Example:
        A custom pseudo-spherical rule of order 3, with ``G(theta) = 1 - ||theta||_3``, only needs to accept
        target probabilities::

            def pseudo_spherical_loss(y_prob, y_true):
                norm = torch.sum(y_prob**3, dim=-1, keepdim=True) ** (1 / 3)
                return torch.sum(y_true * (1.0 - y_prob**2 / norm**2), dim=-1)

            decomposition = SecondOrderScoringRuleDecomposition(sample, pseudo_spherical_loss)
    """

    distribution: SecondOrderDistributionLike
    scoring_rule: ScoringRule

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
