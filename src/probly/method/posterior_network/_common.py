"""Shared definitions for the Posterior Network method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, override, runtime_checkable

from probly.predictor import predict
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import (
    AleatoricEpistemicDecomposition,
    CachingDecomposition,
)
from probly.quantification.measure.distribution import max_probability_complement_of_expected, vacuity
from probly.representation.distribution._common import SecondOrderDistribution
from probly.representation.representation import Representation
from probly.representer import Representer, representer
from probly.transformation.posterior_network import PosteriorNetworkPredictor

if TYPE_CHECKING:
    from probly.quantification.measure.distribution import SecondOrderDistributionLike


@runtime_checkable
class PosteriorNetworkRepresentation(Representation, Protocol):
    """Pseudo-representation type marking outputs of the Posterior Network method.

    A pure marker protocol used to route :func:`decompose` to a PostNet-specific
    decomposition. Constructed via
    :meth:`PosteriorNetworkRepresentation.register_factory` on the
    :class:`PosteriorNetworkRepresenter`'s ``represent`` method, which marks the
    underlying :class:`DirichletDistribution` instance so it is recognised as a
    PostNet representation by the dispatch chain.
    """


# Register as a virtual subclass of SecondOrderDistribution so the dispatch
# considers the marker more specific than the generic SecondOrderDistribution
# entropy decomposition.
SecondOrderDistribution.register(PosteriorNetworkRepresentation)


class PosteriorNetworkRepresenter[**In](
    Representer[Any, In, "PosteriorNetworkRepresentation", "PosteriorNetworkRepresentation"]
):
    """Representer for Posterior Network predictors.

    Calls :func:`predict` on the wrapped PostNet predictor and marks the
    resulting :class:`DirichletDistribution` as a
    :class:`PosteriorNetworkRepresentation`, which lets :func:`decompose`
    auto-route the output to :class:`PosteriorNetworkDecomposition`.
    """

    @override
    @PosteriorNetworkRepresentation.register_factory
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> PosteriorNetworkRepresentation:
        """Return a marked PostNet representation for a given input."""
        return predict(self.predictor, *args, **kwargs)


representer.register(PosteriorNetworkPredictor, PosteriorNetworkRepresenter)


@decompose.register(PosteriorNetworkRepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class PosteriorNetworkDecomposition[T](
    CachingDecomposition,
    AleatoricEpistemicDecomposition[T, T],
):
    """Decomposition based on the Posterior Network.

    Implements the two uncertainty scoring functions formally proposed by
    PostNet :cite:`charpentierPosteriorNetwork2020` in its Sec. 4 ("Metrics"):

    - Aleatoric uncertainty: ``1 - max_c (alpha_c / alpha_0)``, the
      max-probability complement of the predictive distribution. The paper
      reports this as the *aleatoric confidence* score ``max_c p_bar_c``
      ("Alea. Conf" / "OOD Alea" in every results table); this implementation
      flips it to a complement so that high values indicate high uncertainty.
    - Epistemic uncertainty: ``K / alpha_0``, the vacuity introduced by
      :cite:`sensoyEvidentialDeepLearning2018`. This is monotonically
      equivalent to the paper's *"OOD Epist"* score ``alpha_0`` while being
      bounded in ``(0, 1]``.

    The paper does not formally propose a "total uncertainty" scoring
    function; the predictive entropy used in Fig. 4 of the paper is also not
    a formal aleatoric metric (it is only used for OOD/OODom histograms).
    Consequently this decomposition has only aleatoric and epistemic slots,
    and no canonical notion (both are equally valid).

    Args:
        distribution: A second-order distribution (e.g. a Dirichlet) or a
            sample from one. The aleatoric component requires a
            ``max_probability_complement_of_expected`` implementation, and
            the epistemic component requires a ``vacuity`` implementation
            registered for the input type.
    """

    distribution: SecondOrderDistributionLike

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty: max-probability complement of the predictive distribution."""
        return max_probability_complement_of_expected(self.distribution)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty: vacuity ``K / alpha_0`` of the posterior."""
        return vacuity(self.distribution)  # ty:ignore[invalid-return-type]


__all__ = [
    "PosteriorNetworkDecomposition",
    "PosteriorNetworkRepresentation",
    "PosteriorNetworkRepresenter",
]
