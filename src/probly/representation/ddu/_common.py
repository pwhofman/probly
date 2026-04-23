"""Shared DDU representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.representation.representation import Representation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from probly.representation.distribution._common import CategoricalDistribution


@runtime_checkable
class DDURepresentation(Representation, Protocol):
    """Representation of a DDU model output.

    Holds the two quantities needed for uncertainty quantification:
    softmax probabilities (aleatoric) and density vectors (epistemic).
    """

    @property
    def softmax(self) -> CategoricalDistribution:
        """Softmax probabilities."""

    @property
    def densities(self) -> Iterable:
        """Density vectors."""


@flexdispatch
def create_ddu_representation(softmax: CategoricalDistribution, densities: Iterable) -> DDURepresentation:
    """Create a DDU representation from a softmax distribution and density vector."""
    msg = (
        f"No DDU representation factory registered for softmax type {type(softmax)} and density type {type(densities)}"
    )
    raise NotImplementedError(msg)
