"""Decomposition of second-order uncertainty for an arbitrary scoring rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from flextype import flexdispatch

from probly.metrics import zero_one_loss_per_instance
from probly.quantification.decomposition.decomposition import AdditiveDecomposition, ConstantTotalDecomposition
from probly.quantification.decomposition.zero_one._common import SecondOrderZeroOneDecomposition
from probly.quantification.measure.distribution._common import (
    expected_generalized_entropy,
    generalized_entropy_of_expected,
)
from probly.representation.distribution import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    DirichletDistribution,
)
from probly.representation.sample import create_sample

if TYPE_CHECKING:
    from probly.metrics import ScoringRule
    from probly.quantification.decomposition.decomposition import Decomposition
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


def _name(loss: ScoringRule) -> str:
    """Name a scoring rule in an error message."""
    return getattr(loss, "__name__", type(loss).__name__)


@flexdispatch
def scoring_rule_decomposition(representation: object, loss: ScoringRule) -> Decomposition:
    """Decompose the uncertainty of a representation as induced by a scoring rule.

    The scoring rule is the task loss, e.g. of :class:`~probly.selective_prediction.SelectivePredictor`. Methods can
    register this function for their own representation types.

    - A sample of categorical distributions: :class:`SecondOrderZeroOneDecomposition` for the zero-one loss, else
      :class:`SecondOrderScoringRuleDecomposition`.
    - A single categorical distribution: the expected loss of the best prediction as total uncertainty only.
    - A Dirichlet distribution: the zero-one decomposition; other scoring rules are not supported.

    Args:
        representation: The representation to decompose.
        loss: The scoring rule.

    Returns:
        The decomposition of the representation.

    Raises:
        NotImplementedError: If the scoring rule does not apply to the type of ``representation``.
    """
    msg = (
        f"{_name(loss)} is not supported for {type(representation).__name__}; no scoring rule induces a "
        "decomposition of it. Use quantify for its default decomposition, e.g. in a selective predictor "
        "notion=lambda r: quantify(r).epistemic."
    )
    raise NotImplementedError(msg)


@scoring_rule_decomposition.register(CategoricalDistributionSample)
def _(representation: CategoricalDistributionSample, loss: ScoringRule) -> Decomposition:
    if loss is zero_one_loss_per_instance:
        # Unlike total minus aleatoric, its epistemic part is exactly zero where all members agree.
        return SecondOrderZeroOneDecomposition(representation)
    return SecondOrderScoringRuleDecomposition(representation, loss)


@scoring_rule_decomposition.register(CategoricalDistribution)
def _(representation: CategoricalDistribution, loss: ScoringRule) -> Decomposition:
    # A sample with one member reuses the backend implementations.
    return ConstantTotalDecomposition(generalized_entropy_of_expected(create_sample([representation]), loss))


@scoring_rule_decomposition.register(DirichletDistribution)
def _(representation: DirichletDistribution, loss: ScoringRule) -> Decomposition:
    if loss is not zero_one_loss_per_instance:
        msg = f"{_name(loss)} is not supported for Dirichlet distributions; use zero_one_loss_per_instance."
        raise NotImplementedError(msg)
    return SecondOrderZeroOneDecomposition(representation)
