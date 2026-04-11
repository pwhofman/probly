"""Base class for uncertainty quantification methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from probly.representation.representation import Representation


class QuantificationResult(Protocol):
    """Protocol for uncertainty quantifications."""


class Quantifier[R: Representation, Q: QuantificationResult](Protocol):
    """Protocol for uncertainty quantification methods."""

    def __call__(self, representation: R) -> Q:
        """Quantify the uncertainty of a given representation."""


@lazydispatch
def quantify(representation: Representation) -> QuantificationResult:
    """Generic quantify function."""
    msg = f"No quantify function registered for type {type(representation)}"
    raise NotImplementedError(msg)
