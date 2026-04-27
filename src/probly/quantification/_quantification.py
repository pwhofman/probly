"""Base class for uncertainty quantification methods."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.quantification.decomposition.decomposition import Decomposition
    from probly.quantification.measure.measure import MeasureResult
    from probly.representation.representation import Representation

_from_missing_call = ContextVar("from_missing_call", default=False)


class QuantificationResult(Protocol):
    """Protocol for uncertainty quantifications."""


class Quantifier[R: Representation, Q: QuantificationResult](Protocol):
    """Protocol for uncertainty quantification methods."""

    def __call__(self, representation: R) -> Q:
        """Quantify the uncertainty of a given representation."""


@flexdispatch
def decompose(representation: Representation, *args: Any, **kwargs: Any) -> Decomposition:  # noqa: ANN401
    """Decomposes the uncertainty of a given representation."""
    if _from_missing_call.get():
        msg = f"No decompose function registered for type {type(representation)}"
        raise NotImplementedError(msg)

    tok = _from_missing_call.set(True)
    try:
        uncertainty = measure(representation, *args, **kwargs)
    except NotImplementedError as e:
        msg = f"No decompose function registered for type {type(representation)}"
        raise NotImplementedError(msg) from e
    finally:
        _from_missing_call.reset(tok)

    from probly.quantification.decomposition.decomposition import ConstantTotalDecomposition  # noqa: PLC0415

    return ConstantTotalDecomposition(uncertainty)


@flexdispatch
def measure(representation: Representation, *args: Any, **kwargs: Any) -> MeasureResult:  # noqa: ANN401
    """Measures the uncertainty of a given representation."""
    if _from_missing_call.get():
        msg = f"No measure function registered for type {type(representation)}"
        raise NotImplementedError(msg)

    tok = _from_missing_call.set(True)
    try:
        decomposition = decompose(representation, *args, **kwargs)
    except NotImplementedError as e:
        msg = f"No measure function registered for type {type(representation)}"
        raise NotImplementedError(msg) from e
    finally:
        _from_missing_call.reset(tok)
    try:
        return decomposition["total"]
    except KeyError as e:
        msg = f"Decomposition for type {type(representation)} does not have a (default) total uncertainty notion."
        raise NotImplementedError(msg) from e


@flexdispatch
def quantify(representation: Representation, *args: Any, **kwargs: Any) -> QuantificationResult:  # noqa: ANN401
    """Quantifies the uncertainty of a given representation.

    Usually tries to decompose the uncertainty of the representation.
    Representations can however opt to provide an entirely custom notion of uncertainty quantification.
    """
    try:
        return decompose(representation, *args, **kwargs)
    except NotImplementedError as e:
        msg = f"No quantify function registered for type {type(representation)}"
        raise NotImplementedError(msg) from e
