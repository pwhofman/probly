"""Shared definitions for the evidential classification method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, override, runtime_checkable

from probly.predictor import predict
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import CachingDecomposition, EpistemicDecomposition
from probly.quantification.measure.distribution import vacuity
from probly.representation.distribution import DirichletDistribution
from probly.representation.distribution._common import SecondOrderDistribution
from probly.representation.representation import Representation
from probly.representer import Representer, representer
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


@runtime_checkable
class EvidentialClassificationRepresentation(Representation, Protocol):
    """Pseudo-representation type marking outputs of the evidential classification method.

    A pure marker protocol used to route :func:`decompose` to an EDL-specific
    decomposition. Constructed via
    :meth:`EvidentialClassificationRepresentation.register_factory` on the
    :class:`EvidentialClassificationRepresenter`'s ``represent`` method, which marks
    the underlying :class:`DirichletDistribution` instance so it is recognised as
    an EDL representation by the dispatch chain.
    """


# Register as a virtual subclass of SecondOrderDistribution so the dispatch
# considers the marker more specific than the generic SecondOrderDistribution
# entropy decomposition.
SecondOrderDistribution.register(EvidentialClassificationRepresentation)


class EvidentialClassificationRepresenter[**In](
    Representer[Any, In, "EvidentialClassificationRepresentation", "EvidentialClassificationRepresentation"]
):
    """Representer for evidential classification predictors.

    Calls :func:`predict` on the wrapped EDL predictor and marks the
    resulting :class:`DirichletDistribution` as an
    :class:`EvidentialClassificationRepresentation`, which lets :func:`decompose`
    auto-route the output to :class:`EvidentialClassificationDecomposition`.
    """

    @override
    @EvidentialClassificationRepresentation.register_factory
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> EvidentialClassificationRepresentation:
        """Return a marked EDL representation for a given input."""
        return predict(self.predictor, *args, **kwargs)


representer.register(EvidentialClassificationPredictor, EvidentialClassificationRepresenter)


@decompose.register(EvidentialClassificationRepresentation)
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
    "EvidentialClassificationRepresentation",
    "EvidentialClassificationRepresenter",
    "evidential_classification",
]
