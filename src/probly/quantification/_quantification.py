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
def measure_atomic(representation: Representation, *args: Any, **kwargs: Any) -> MeasureResult:  # noqa: ANN401
    """Measures the uncertainty of a given representation without attempting to decompose it."""
    msg = f"No measure_atomic function registered for type {type(representation)}"
    raise NotImplementedError(msg)


@flexdispatch
def measure(representation: Representation, *args: Any, **kwargs: Any) -> MeasureResult:  # noqa: ANN401
    """Measures the uncertainty of a given representation.

    By default, this tries to decompose the uncertainty of the representation and returns the canonical notion of
    uncertainty of the decomposition. Normally, the canonical notion is the total uncertainty.
    Some representations may however provide a different or no canonical notion at all.
    """
    if _from_missing_call.get():
        return measure_atomic(representation, *args, **kwargs)

    tok = _from_missing_call.set(True)
    try:
        decompose_impl = decompose.dispatch(type(representation), registry_meta_lookup=representation)
        default_decompose_impl = decompose.dispatch(object)
        # If no explicit decompose implementation is registered,
        # attempt to measure uncertainty atomically without decomposition:
        if decompose_impl is default_decompose_impl:
            return measure_atomic(representation, *args, **kwargs)
        decomposition = decompose_impl(representation, *args, **kwargs)
    finally:
        _from_missing_call.reset(tok)

    return decomposition.get_canonical()


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
