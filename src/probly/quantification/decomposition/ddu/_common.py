"""Quantification of DUQ representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from flextype import flexdispatch
from probly.quantification._quantification import decompose
from probly.quantification.decomposition import AleatoricEpistemicDecomposition, CachingDecomposition
from probly.quantification.measure.distribution import entropy
from probly.representation.ddu import DDURepresentation

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike


@flexdispatch
def ddu_epistemic_uncertainty(representation: DDURepresentation) -> ArrayLike:
    r"""Compute the DDU epistemic uncertainty scores based on the fitted GMM.

    Args:
        representation: DDU representation produced by a DDU predictor.

    Returns:
        Per-sample uncertainty scores of shape ``(...,)``.
    """
    msg = f"DDU epistemic uncertainty is not implemented for representations of type {type(representation)}"
    raise NotImplementedError(msg)


@decompose.register(DDURepresentation)
@dataclass(frozen=True, slots=True, repr=True)
class DDUDecomposition[T](CachingDecomposition, AleatoricEpistemicDecomposition[T, T]):
    """Base class for entropy-based decomposition methods."""

    representation: DDURepresentation

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty of the decomposition."""
        return entropy(self.representation.softmax)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty of the decomposition."""
        return ddu_epistemic_uncertainty(self.representation)  # ty:ignore[invalid-return-type]
