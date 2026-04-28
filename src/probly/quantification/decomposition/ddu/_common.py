"""DDU-specific uncertainty decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from flextype import flexdispatch
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import AleatoricEpistemicDecomposition, CachingDecomposition
from probly.quantification.measure.distribution import LogBase, entropy
from probly.representation.ddu import DDURepresentation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from probly.representation.array_like import ArrayLike


@flexdispatch
def negative_log_density(densities: Iterable) -> ArrayLike:
    """Convert DDU log-density scores to an epistemic uncertainty score."""
    msg = f"Negative log density is not supported for densities of type {type(densities)}."
    raise NotImplementedError(msg)


@decompose.register(DDURepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class DDUDensityDecomposition[T](CachingDecomposition, AleatoricEpistemicDecomposition[T, T]):
    """DDU decomposition into softmax entropy and negative feature log density.

    DDU does not define an additive total uncertainty. The aleatoric component is
    the entropy of the softmax distribution, while the epistemic component is a
    monotone uncertainty score derived from the feature-space density.
    """

    representation: DDURepresentation
    base: LogBase = None

    @override
    @property
    def _aleatoric(self) -> T:
        """The aleatoric uncertainty of the decomposition."""
        return entropy(self.representation.softmax, base=self.base)  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> T:
        """The epistemic uncertainty of the decomposition."""
        return negative_log_density(self.representation.densities)  # ty:ignore[invalid-return-type]
