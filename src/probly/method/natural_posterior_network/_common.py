"""Shared definitions for the Natural Posterior Network method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, override, runtime_checkable

from probly.predictor import predict
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import (
    AleatoricEpistemicTotalDecomposition,
    CachingDecomposition,
)
from probly.quantification.measure.distribution import (
    entropy,
    entropy_of_expected_predictive_distribution,
    vacuity,
)
from probly.representation.distribution._common import SecondOrderDistribution
from probly.representation.representation import Representation
from probly.representer import Representer, representer
from probly.transformation.natural_posterior_network import NaturalPosteriorNetworkPredictor

if TYPE_CHECKING:
    from probly.quantification.measure.distribution import SecondOrderDistributionLike
    from probly.quantification.measure.distribution._common import LogBase


@runtime_checkable
class NaturalPosteriorNetworkRepresentation(Representation, Protocol):
    """Pseudo-representation type marking outputs of the Natural Posterior Network method.

    A pure marker protocol used to route :func:`decompose` to a NatPN-specific
    decomposition. Constructed via
    :meth:`NaturalPosteriorNetworkRepresentation.register_factory` on the
    :class:`NaturalPosteriorNetworkRepresenter`'s ``represent`` method, which marks
    the underlying :class:`DirichletDistribution` instance so it is recognised as
    a NatPN representation by the dispatch chain.
    """


# Register as a virtual subclass of SecondOrderDistribution so the dispatch
# considers the marker more specific than the generic SecondOrderDistribution
# entropy decomposition.
SecondOrderDistribution.register(NaturalPosteriorNetworkRepresentation)


class NaturalPosteriorNetworkRepresenter[**In](
    Representer[Any, In, "NaturalPosteriorNetworkRepresentation", "NaturalPosteriorNetworkRepresentation"]
):
    """Representer for Natural Posterior Network predictors.

    Calls :func:`predict` on the wrapped NatPN predictor and marks the
    resulting :class:`DirichletDistribution` as a
    :class:`NaturalPosteriorNetworkRepresentation`, which lets :func:`decompose`
    auto-route the output to :class:`NaturalPosteriorDecomposition`.
    """

    @override
    @NaturalPosteriorNetworkRepresentation.register_factory
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> NaturalPosteriorNetworkRepresentation:
        """Return a marked NatPN representation for a given input."""
        return predict(self.predictor, *args, **kwargs)


representer.register(NaturalPosteriorNetworkPredictor, NaturalPosteriorNetworkRepresenter)


@decompose.register(NaturalPosteriorNetworkRepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class NaturalPosteriorDecomposition[T](
    CachingDecomposition,
    AleatoricEpistemicTotalDecomposition[T, T, T],
):
    """Decomposition based on the Natural Posterior Network.

    Implements the three uncertainty quantities formally proposed by NatPN
    :cite:`charpentierNaturalPosteriorNetwork2022` in Appendix E
    ("Formulae for Uncertainty Estimates"). For a Dirichlet posterior
    Dir(alpha) with K classes and alpha_0 = sum_c alpha_c, the components are:

    - Total uncertainty: ``H[Dir(alpha)]``, the differential entropy of the
      posterior distribution. Appendix E: *"The entropy of the posterior
      distribution Q(theta | chi^post, n^post) was used to estimate the
      predictive uncertainty"*. The paper writes this as "predictive
      uncertainty"; we map it to the standard UQ "total uncertainty" slot.
    - Aleatoric uncertainty: ``H[Cat(alpha / alpha_0)]``, the entropy of the
      target distribution evaluated at the posterior mean. Appendix E:
      *"The entropy of the target distribution P(y | theta) was used to
      estimate the aleatoric uncertainty"*. For categorical y the target
      distribution at chi^post = alpha / alpha_0 is Cat(alpha / alpha_0).
    - Epistemic uncertainty: ``K / alpha_0``, the vacuity introduced by
      :cite:`sensoyEvidentialDeepLearning2018`. Appendix E uses the predicted
      evidence ``n^post = alpha_0`` directly; vacuity is its monotone
      transform into ``(0, 1]`` (high = uncertain).

    The decomposition is non-additive: ``total != aleatoric + epistemic`` in
    general, since the three quantities are defined on different spaces.

    Args:
        distribution: A second-order distribution (e.g. a Dirichlet) or a
            sample from one. Total and aleatoric components require
            ``entropy`` / ``entropy_of_expected_predictive_distribution``
            implementations, while the epistemic component requires a
            ``vacuity`` implementation registered for the input type.
        base: Logarithm base used for the entropy-based components. ``None``
            uses the natural logarithm. The vacuity is dimensionless and is
            unaffected by ``base``.
    """

    distribution: SecondOrderDistributionLike
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        """The total / predictive uncertainty: differential entropy of the posterior distribution."""
        return entropy(self.distribution, base=self.base)

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty: entropy of the target distribution at the posterior mean."""
        return entropy_of_expected_predictive_distribution(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty: vacuity ``K / alpha_0`` of the posterior."""
        return vacuity(self.distribution)  # ty:ignore[invalid-return-type]


__all__ = [
    "NaturalPosteriorDecomposition",
    "NaturalPosteriorNetworkRepresentation",
    "NaturalPosteriorNetworkRepresenter",
]
