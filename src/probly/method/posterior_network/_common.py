"""Shared definitions for the Posterior Network method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification.decomposition.decomposition import (
    AleatoricEpistemicTotalDecomposition,
    CachingDecomposition,
)
from probly.quantification.measure.distribution import (
    entropy_of_expected_predictive_distribution,
    max_probability_complement_of_expected,
    vacuity,
)

if TYPE_CHECKING:
    from probly.quantification.measure.distribution import SecondOrderDistributionLike
    from probly.quantification.measure.distribution._common import LogBase


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class PosteriorNetworkDecomposition[T](
    CachingDecomposition,
    AleatoricEpistemicTotalDecomposition[T, T, T],
):
    """Decomposition based on the Posterior Network.

    Implements the uncertainty scoring functions of PostNet
    :cite:`charpentierPosteriorNetwork2020`. For a Dirichlet posterior
    Dir(alpha) with K classes and alpha_0 = sum_c alpha_c, the components are:

    - Total uncertainty: ``1 - max_c (alpha_c / alpha_0)``, the max-probability
      complement of the predictive distribution. This is the score the paper
      uses for confidence calibration / misclassification detection (the
      "Alea. Conf" column in every results table), reported as the confidence
      ``max_c p_bar_c`` and converted here to an uncertainty.
    - Aleatoric uncertainty: ``H[Cat(alpha / alpha_0)]``, the entropy of the
      aleatoric distribution. Used in Fig. 4 of the paper to visualize the
      separation between in-distribution, out-of-distribution and out-of-domain
      inputs.
    - Epistemic uncertainty: ``K / alpha_0``, the vacuity introduced by
      :cite:`sensoyEvidentialDeepLearning2018`. This is monotonically equivalent
      (i.e. produces the same ranking) to the paper's "OOD Epist" score
      ``alpha_0`` while being bounded in ``(0, 1]``.

    The decomposition is non-additive: ``total != aleatoric + epistemic`` in
    general, since the three quantities are different scoring functions rather
    than an information-theoretic split. The total uncertainty is the score the
    paper validates as the best at separating correct from incorrect
    predictions, hence its choice as the canonical notion.

    Args:
        distribution: A second-order distribution (e.g. a Dirichlet) or a sample
            from one. The total component requires a
            ``max_probability_complement_of_expected`` implementation, the
            aleatoric component requires an
            ``entropy_of_expected_predictive_distribution`` implementation, and
            the epistemic component requires a ``vacuity`` implementation
            registered for the input type.
        base: Logarithm base used for the aleatoric (entropy-based) component.
            ``None`` uses the natural logarithm. The total uncertainty is a
            probability complement, and the epistemic uncertainty (vacuity) is
            dimensionless; both are unaffected by ``base``.
    """

    distribution: SecondOrderDistributionLike
    base: LogBase = None

    @override
    @property
    def _total(self) -> T:
        """The total uncertainty: max-probability complement of the predictive distribution."""
        return max_probability_complement_of_expected(self.distribution)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty: entropy of the predictive distribution at the posterior mean."""
        return entropy_of_expected_predictive_distribution(self.distribution, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty: vacuity ``K / alpha_0`` of the posterior."""
        return vacuity(self.distribution)  # ty:ignore[invalid-return-type]


__all__ = ["PosteriorNetworkDecomposition"]
