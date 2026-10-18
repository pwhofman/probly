"""Decomposition of second-order uncertainty for an arbitrary scoring rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from flextype import flexdispatch

from probly.quantification.decomposition.decomposition import AdditiveDecomposition, ConstantTotalDecomposition
from probly.quantification.decomposition.zero_one._common import SecondOrderZeroOneDecomposition
from probly.quantification.measure.distribution._common import (
    expected_generalized_entropy,
    generalized_entropy_of_expected,
)
from probly.quantification.scoring_rule import ScoringRule, ZeroOneLoss
from probly.representation.distribution import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    DirichletDistribution,
)
from probly.representation.sample import create_sample

if TYPE_CHECKING:
    from probly.quantification.decomposition.decomposition import Decomposition
    from probly.quantification.measure.distribution._common import SecondOrderDistributionLike


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class SecondOrderScoringRuleDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Total/aleatoric/epistemic decomposition induced by a scoring rule.

    For a second-order categorical distribution ``Q`` with Bayesian model average
    ``theta_bar = E[theta]`` and generalized entropy ``G(theta) = <theta, loss(theta)>``:

    - Total uncertainty: ``G(theta_bar)``.
    - Aleatoric uncertainty: ``E_{theta ~ Q}[G(theta)]``.
    - Epistemic uncertainty: ``total - aleatoric`` (a Jensen gap, non-negative when the rule is proper).

    The decomposition is additive: ``total == aleatoric + epistemic``.
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
        f"{type(loss).__name__} is not supported for {type(representation).__name__}; "
        "omit loss to use the default decomposition of this representation, or pass loss=None."
    )
    raise NotImplementedError(msg)


@scoring_rule_decomposition.register(CategoricalDistributionSample)
def _(representation: CategoricalDistributionSample, loss: ScoringRule) -> Decomposition:
    if isinstance(loss, ZeroOneLoss):
        # Unlike total minus aleatoric, its epistemic part is exactly zero where all members agree.
        return SecondOrderZeroOneDecomposition(representation)
    return SecondOrderScoringRuleDecomposition(representation, loss)


@scoring_rule_decomposition.register(CategoricalDistribution)
def _(representation: CategoricalDistribution, loss: ScoringRule) -> Decomposition:
    # A sample with one member reuses the backend implementations.
    return ConstantTotalDecomposition(generalized_entropy_of_expected(create_sample([representation]), loss))


@scoring_rule_decomposition.register(DirichletDistribution)
def _(representation: DirichletDistribution, loss: ScoringRule) -> Decomposition:
    if not isinstance(loss, ZeroOneLoss):
        msg = f"{type(loss).__name__} is not supported for Dirichlet distributions; use ZeroOneLoss() or loss=None."
        raise NotImplementedError(msg)
    return SecondOrderZeroOneDecomposition(representation)
