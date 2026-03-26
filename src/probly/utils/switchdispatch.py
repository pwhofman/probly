"""An extensible switch dispatch mechanism.

Akin to the builtin functools.singledispatch mechanism,
but with an equality-based dispatcher.
"""

from __future__ import annotations

from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Concatenate, overload

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class switchdispatch[T, **In, Out]:  # noqa: N801
    """A switch dispatch decorator.

    Similar to functools.singledispatch, but dispatches based on equality
    rather than type.

    Example:
        >>> @switchdispatch
        >>> def func(x):
        >>>     return "default"
        >>>
        >>> @func.register(1)
        >>> def _(x):
        >>>     return "one"
        >>>
        >>> @func.register(2)
        >>> def _(x):
        >>>     return "two"
        >>>
        >>> print(func(1))  # Output: "one"
        >>> print(func(2))  # Output: "two"
        >>> print(func(3))  # Output: "default"
    """

    def __init__(self, func: Callable[Concatenate[T, In], Out]) -> None:
        """Initialize the switchdispatch with the default function."""
        self._func = func
        self._registry: dict[object, Callable[Concatenate[T, *In], Out]] = {}
        update_wrapper(self, func, updated=())

    @overload
    def register(self, key: T) -> Callable[[Callable[Concatenate[T, In], Out]], Callable[Concatenate[T, In], Out]]: ...

    @overload
    def register(self, key: T, f: Callable[Concatenate[T, In], Out]) -> Callable[Concatenate[T, In], Out]: ...

    def register(
        self,
        key: object,
        f: Callable[Concatenate[T, In], Out] | None = None,
    ) -> (
        Callable[Concatenate[T, In], Out]
        | Callable[[Callable[Concatenate[T, In], Out]], Callable[Concatenate[T, In], Out]]
    ):
        """Register a new function for the given key."""
        return self.multi_register([key], f)  # ty:ignore[invalid-argument-type]

    @overload
    def multi_register(
        self, keys: Iterable[object]
    ) -> Callable[[Callable[Concatenate[T, In], Out]], Callable[Concatenate[T, In], Out]]: ...

    @overload
    def multi_register(
        self, keys: Iterable[object], f: Callable[Concatenate[T, In], Out]
    ) -> Callable[Concatenate[T, In], Out]: ...

    def multi_register(
        self,
        keys: Iterable[object],
        f: Callable[Concatenate[T, In], Out] | None = None,
    ) -> (
        Callable[Concatenate[T, In], Out]
        | Callable[[Callable[Concatenate[T, In], Out]], Callable[Concatenate[T, In], Out]]
    ):
        """Register a new function for the given keys."""
        if f is not None:
            for key in keys:
                self._registry[key] = f
            return f

        def wrapper(f: Callable[Concatenate[T, In], Out]) -> Callable[Concatenate[T, In], Out]:
            for key in keys:
                self._registry[key] = f
            return f

        return wrapper

    def __call__(self, arg: object, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Call the appropriate function based on the argument."""
        func_impl = self._registry.get(arg, self._func)
        return func_impl(arg, *args, **kwargs)
