"""Shared definitions for the Posterior Network method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification.decomposition.decomposition import (
    AleatoricEpistemicDecomposition,
    CachingDecomposition,
)
from probly.quantification.measure.distribution import max_probability_complement_of_expected, vacuity

if TYPE_CHECKING:
    from probly.quantification.measure.distribution import SecondOrderDistributionLike


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


__all__ = ["PosteriorNetworkDecomposition"]
