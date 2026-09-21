"""Common definitions of credal set measures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload, override

from flextype import Flexdispatch, flexdispatch

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike
    from probly.representation.credal_set._common import CredalSet

type LogBase = float | Literal["normalize"] | None
type _EntropyResult = ArrayLike | tuple[ArrayLike, ArrayLike]


class _EntropyDispatcher(Flexdispatch[..., _EntropyResult]):
    """A flexdispatcher whose result type depends on ``return_distribution``."""

    # Keep the overloads on the callable object so its inherited registration
    # API remains visible alongside the precise return types.
    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: Literal[False] = False,
    ) -> ArrayLike: ...

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: Literal[True],
    ) -> tuple[ArrayLike, ArrayLike]: ...

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: bool,
    ) -> _EntropyResult: ...

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> _EntropyResult:
        """Forward arguments unchanged to the registered implementation."""
        return super().__call__(*args, **kwargs)


@_EntropyDispatcher
def upper_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """Compute the upper entropy of a credal set.

    If ``return_distribution`` is ``True``, returns ``(entropy, distribution)``
    where ``distribution`` is the maximizer of shape ``(..., num_classes)``.
    """
    msg = f"Upper entropy is not supported for credal sets of type {type(credal_set)}."
    raise NotImplementedError(msg)


@_EntropyDispatcher
def lower_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """Compute the lower entropy of a credal set.

    If ``return_distribution`` is ``True``, returns ``(entropy, distribution)``
    where ``distribution`` is the minimizer of shape ``(..., num_classes)``.
    """
    msg = f"Lower entropy is not supported for credal sets of type {type(credal_set)}."
    raise NotImplementedError(msg)


@flexdispatch
def generalized_hartley(credal_set: CredalSet, base: LogBase = None) -> ArrayLike:
    """Compute the generalized Hartley measure of a credal set."""
    msg = f"Generalized Hartley measure is not supported for credal sets of type {type(credal_set)}."
    raise NotImplementedError(msg)
