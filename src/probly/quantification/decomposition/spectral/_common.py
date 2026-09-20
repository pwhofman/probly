"""Common code for spectral uncertainty decompositions of embedding representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.spectral import conditional_spectral_entropy, spectral_entropy
from probly.representation.embedding._common import EmbeddingSampleSample

if TYPE_CHECKING:
    from probly.representation.embedding._common import Embedding


@decompose.register(EmbeddingSampleSample)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class SpectralDecomposition[T](AdditiveDecomposition[T, T, T]):
    """Spectral decomposition into total, aleatoric, and epistemic uncertainty :cite:`walhaFineGrainedUncertainty2026`.

    The outer sample holds the groups, such as clarifications of a prompt, and each inner sample the responses within
    a group. The total uncertainty is the Von Neumann entropy of the kernel over all responses, the aleatoric
    uncertainty is the mean Von Neumann entropy within a group, and the epistemic uncertainty is their difference,
    the Holevo information.
    """

    embeddings: EmbeddingSampleSample[Embedding[T]]
    gamma: float = 1.0
    normalized: bool = True
    eps: float = 1e-12

    @override
    @property
    def _total(self) -> T:
        """The total spectral uncertainty."""
        return spectral_entropy(self.embeddings, gamma=self.gamma, normalized=self.normalized, eps=self.eps)

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric spectral uncertainty."""
        return conditional_spectral_entropy(self.embeddings, gamma=self.gamma, normalized=self.normalized, eps=self.eps)
