"""Distance-based (Wasserstein) decomposition of second-order uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from probly.quantification.decomposition.decomposition import (
    AleatoricEpistemicTotalDecomposition,
    CachingDecomposition,
)
from probly.quantification.measure.distribution import (
    DEFAULT_NUM_SAMPLES,
    expected_max_probability_complement,
    max_probability_complement_of_expected,
    min_expected_total_variation,
)
from probly.representation.sample import Sample

if TYPE_CHECKING:
    from probly.quantification.measure.distribution import SecondOrderDistributionLike


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class SecondOrderWassersteinDecomposition[T](AleatoricEpistemicTotalDecomposition[T, T, T], CachingDecomposition):
    """Distance-based decomposition of second-order uncertainty :cite:`saleSecondOrder2024`.

    The three Wasserstein distance-based measures for a second-order distribution ``Q`` (a sample
    of categorical distributions or a Dirichlet distribution):

    - Total uncertainty: ``1 - max_y E_Q[p(y)]``.
    - Aleatoric uncertainty: ``1 - E_Q[max_y p(y)]``.
    - Epistemic uncertainty: ``1/2 min_q E_{p ~ Q}[||p - q||_1]``.

    The measures are defined independently, so the decomposition is not additive
    (``total != aleatoric + epistemic`` in general), and each lies in ``[0, (K - 1) / K]`` for
    ``K`` classes.

    Attributes:
        distribution: The second-order distribution to decompose.
        num_samples: Monte-Carlo draws for the aleatoric and epistemic uncertainties of a parametric
            distribution such as a Dirichlet. Ignored for a sample.
        generator: Optional ``numpy.random.Generator`` for reproducible draws (numpy backend only).
            The torch backend seeds via ``torch.manual_seed``; a generator passed there is ignored
            with a warning.
    """

    distribution: SecondOrderDistributionLike
    num_samples: int = DEFAULT_NUM_SAMPLES
    generator: object | None = field(default=None, compare=False)

    @property
    def _monte_carlo_kwargs(self) -> dict[str, Any]:
        """Sampling controls for the aleatoric and epistemic measures.

        They apply only to a parametric distribution that must be sampled, such as a Dirichlet. A
        sample already carries its draws, so nothing is passed.
        """
        if isinstance(self.distribution, Sample):
            return {}
        return {"num_samples": self.num_samples, "generator": self.generator}

    @override
    @property
    def _total(self) -> T:
        """The total uncertainty of the decomposition."""
        return max_probability_complement_of_expected(self.distribution)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty of the decomposition."""
        return expected_max_probability_complement(self.distribution, **self._monte_carlo_kwargs)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty of the decomposition."""
        return min_expected_total_variation(self.distribution, **self._monte_carlo_kwargs)  # ty:ignore[invalid-return-type]
