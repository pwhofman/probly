"""Shared definitions for the evidential classification method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from probly.quantification.decomposition.decomposition import CachingDecomposition, EpistemicDecomposition
from probly.quantification.measure.distribution import vacuity
from probly.representation.distribution import DirichletDistribution
from probly.transformation.dirichlet_clipped_exp_one_activation import (
    DirichletClippedExpOneActivationPredictor,
    dirichlet_clipped_exp_one_activation,
)

if TYPE_CHECKING:
    from probly.quantification.measure.distribution import SecondOrderDistributionLike


@runtime_checkable
class EvidentialClassificationPredictor[**In, Out: DirichletDistribution](
    DirichletClippedExpOneActivationPredictor[In, Out], Protocol
):
    """A predictor routed through the evidential classification method API."""


evidential_classification = EvidentialClassificationPredictor.register_factory(
    dirichlet_clipped_exp_one_activation,
)


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class EvidentialClassificationDecomposition[T](CachingDecomposition, EpistemicDecomposition[T]):
    """Decomposition based on Evidential Deep Learning for classification.

    Implements the single uncertainty quantity of EDL
    :cite:`sensoyEvidentialDeepLearning2018`. For a Dirichlet distribution
    Dir(alpha) with K classes and S = sum_c alpha_c, the only uncertainty
    measure formally proposed by the paper is the vacuity:

    - Epistemic uncertainty: ``K / S``, the "uncertainty mass" or vacuity from
      Dempster-Shafer / subjective-logic theory (Eq. 1 of the paper). Sensoy
      et al. explicitly call this the epistemic uncertainty of the
      classification (Sec. 3 of the paper).

    The paper does not propose any aleatoric measure or aleatoric/epistemic
    decomposition; the subjective-opinion framework only splits between belief
    mass ``b_c = (alpha_c - 1) / S`` per class and the single vacuity term.
    Consequently this decomposition has only an epistemic slot, and its
    canonical notion is epistemic uncertainty.

    Args:
        distribution: A second-order distribution (e.g. a Dirichlet) or a
            sample from one. Requires a ``vacuity`` implementation registered
            for the input type.
    """

    distribution: SecondOrderDistributionLike

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty: vacuity ``K / S`` of the Dirichlet."""
        return vacuity(self.distribution)  # ty:ignore[invalid-return-type]


__all__ = [
    "EvidentialClassificationDecomposition",
    "EvidentialClassificationPredictor",
    "evidential_classification",
]
