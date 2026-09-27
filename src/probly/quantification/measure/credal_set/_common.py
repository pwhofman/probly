"""Common definitions of credal set measures."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, overload, override
import warnings

from flextype import Flexdispatch, flexdispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from flextype import LazyType

    from probly.representation.array_like import ArrayLike
    from probly.representation.credal_set._common import CredalSet

type LogBase = float | Literal["normalize"] | None
type Approximate = bool | Literal["auto"]
type _EntropyResult = ArrayLike | tuple[ArrayLike, ArrayLike]

# Probability intervals with at most this many classes get the exact lower entropy.
EXACT_LOWER_ENTROPY_MAX_CLASSES = 14


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


class _LowerEntropyDispatcher(_EntropyDispatcher):
    """An entropy dispatcher that also takes the ``approximate`` option of :func:`lower_entropy`."""

    @overload
    def register_approx[F: Callable[..., Any]](self, cls: LazyType, func: F) -> F: ...

    @overload
    def register_approx[F: Callable[..., Any]](self, cls: LazyType) -> Callable[[F], F]: ...

    def register_approx(self, cls: LazyType, func: Callable | None = None) -> Callable:
        """Register an approximation-only implementation without an ``approximate`` parameter.

        The registered wrapper rejects ``approximate=False`` and consumes ``True`` or
        ``"auto"`` before calling the implementation.

        Args:
            cls: The dispatch type or lazy type specification.
            func: The approximation. If omitted, return a registration decorator.

        Returns:
            The original function, or a decorator that registers and returns it.
        """
        if func is None:
            return lambda implementation: self.register_approx(cls, implementation)

        @wraps(func)
        def approx(
            credal_set: CredalSet, *args: object, approximate: Approximate = "auto", **kwargs: object
        ) -> _EntropyResult:
            if approximate is False:
                msg = (
                    f"The lower entropy of {type(credal_set).__name__} only has an approximate implementation, "
                    "so approximate=False is not supported. Use approximate=True or approximate='auto'."
                )
                raise ValueError(msg)
            return func(credal_set, *args, **kwargs)

        super().register(cls, approx)
        return func

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: Literal[False] = False,
        approximate: Approximate = "auto",
    ) -> ArrayLike: ...

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: Literal[True],
        approximate: Approximate = "auto",
    ) -> tuple[ArrayLike, ArrayLike]: ...

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: bool,
        approximate: Approximate = "auto",
    ) -> _EntropyResult: ...

    @override
    def __call__(self, credal_set: CredalSet, *args: Any, **kwargs: Any) -> _EntropyResult:
        """Check ``approximate`` and forward the arguments to the registered implementation."""
        approximate = kwargs.setdefault("approximate", "auto")
        if approximate not in (True, False, "auto"):
            msg = f"approximate must be True, False or 'auto', got {approximate!r}."
            raise ValueError(msg)
        return super().__call__(credal_set, *args, **kwargs)


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


@_LowerEntropyDispatcher
def lower_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
    approximate: Approximate = "auto",
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """Compute the lower entropy of a credal set.

    If ``return_distribution`` is ``True``, returns ``(entropy, distribution)``
    where ``distribution`` is the minimizer of shape ``(..., num_classes)``.

    ``approximate=False`` requires an exact computation, while ``True`` permits an
    approximation. The default, ``"auto"``, selects the available strategy. Exact-only
    implementations accept all three values and always compute the exact result.
    Approximation-only implementations reject ``False`` with a ``ValueError``.

    For probability intervals and Dirichlet level sets, the exact lower entropy tries every
    extreme point of the set, which is only feasible for up to 14 classes. ``approximate``
    selects how it is computed:

    - ``False``: exactly. More than 14 classes raise a ``ValueError``.
    - ``True``: with a greedy search, which gives an upper bound on the lower entropy.
    - ``"auto"`` (default): exactly for up to 14 classes, and with the greedy search and a warning for
      more classes.

    Distance-based credal sets only have an approximate lower entropy implementation.
    Dirichlet level sets use sampled per-class bounds; their ``approximate`` option
    controls entropy optimization over those bounds, not the sampling approximation.
    """
    msg = f"Lower entropy is not supported for credal sets of type {type(credal_set)}."
    raise NotImplementedError(msg)


def use_approximate_lower_entropy(n_classes: int, approximate: Approximate) -> bool:
    """Return whether the lower entropy of probability intervals with ``n_classes`` classes is approximated.

    Args:
        n_classes: Number of classes of the credal set.
        approximate: The ``approximate`` option of :func:`lower_entropy`.

    Returns:
        ``True`` if the greedy search is used, ``False`` if the lower entropy is exact.

    Raises:
        ValueError: If ``approximate`` is ``False`` and there are more than
            ``EXACT_LOWER_ENTROPY_MAX_CLASSES`` classes.
    """
    if approximate is True:
        return True
    if n_classes <= EXACT_LOWER_ENTROPY_MAX_CLASSES:
        return False
    if approximate == "auto":
        warnings.warn(
            f"The lower entropy of probability intervals with {n_classes} classes is approximated by a "
            f"greedy search, since the exact value is only computed for up to "
            f"{EXACT_LOWER_ENTROPY_MAX_CLASSES} classes. The approximation is an upper bound, so the "
            "aleatoric uncertainty can be too high and the epistemic uncertainty too low.",
            stacklevel=2,
        )
        return True
    msg = (
        f"The exact lower entropy of probability intervals is only computed for up to "
        f"{EXACT_LOWER_ENTROPY_MAX_CLASSES} classes, but this credal set has {n_classes}. Pass "
        "approximate=True to use a greedy search, which gives an upper bound on the lower entropy, "
        "or approximate='auto' to use it only when there are too many classes."
    )
    raise ValueError(msg)


@flexdispatch
def generalized_hartley(credal_set: CredalSet, base: LogBase = None) -> ArrayLike:
    """Compute the generalized Hartley measure of a credal set."""
    msg = f"Generalized Hartley measure is not supported for credal sets of type {type(credal_set)}."
    raise NotImplementedError(msg)
