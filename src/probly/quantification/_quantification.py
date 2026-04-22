"""Base class for uncertainty quantification methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.representation.representation import Representation


class QuantificationResult(Protocol):
    """Protocol for uncertainty quantifications."""


class Quantifier[R: Representation, Q: QuantificationResult](Protocol):
    """Protocol for uncertainty quantification methods."""

    def __call__(self, representation: R) -> Q:
        """Quantify the uncertainty of a given representation."""


@flexdispatch
def quantify(representation: Representation, *args: Any, **kwargs: Any) -> QuantificationResult:  # noqa: ANN401
    """Generic quantify function."""
    msg = f"No quantify function registered for type {type(representation)}"
    raise NotImplementedError(msg)
