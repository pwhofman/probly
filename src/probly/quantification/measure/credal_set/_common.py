"""Common definitions of credal set measures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike
    from probly.representation.credal_set._common import CredalSet

type LogBase = float | Literal["normalize"] | None


@overload
def upper_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    n_jobs: int | None = None,
    *,
    return_distribution: Literal[False] = False,
) -> ArrayLike: ...
@overload
def upper_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    n_jobs: int | None = None,
    *,
    return_distribution: Literal[True],
) -> tuple[ArrayLike, ArrayLike]: ...
@flexdispatch
def upper_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    n_jobs: int | None = None,
    *,
    return_distribution: bool = False,
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """Compute the upper entropy of a credal set.

    If ``return_distribution`` is ``True``, returns ``(entropy, distribution)``
    where ``distribution`` is the maximizer of shape ``(..., num_classes)``.
    """
    msg = f"Upper entropy is not supported for credal sets of type {type(credal_set)}."
    raise NotImplementedError(msg)


@overload
def lower_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    n_jobs: int | None = None,
    *,
    return_distribution: Literal[False] = False,
) -> ArrayLike: ...
@overload
def lower_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    n_jobs: int | None = None,
    *,
    return_distribution: Literal[True],
) -> tuple[ArrayLike, ArrayLike]: ...
@flexdispatch
def lower_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    n_jobs: int | None = None,
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
