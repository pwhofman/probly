"""An extensible switch dispatch mechanism.

Akin to the builtin functools.singledispatch mechanism,
but with an equality-based dispatcher.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from functools import update_wrapper
from typing import TYPE_CHECKING, Concatenate, Protocol, overload, override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import NotImplementedType


class _RegistrationSink[K, V](Protocol):
    """The contravariant write interface used to constrain registered values."""

    def _register_value(self, key: K, value: V) -> None: ...


class _Registration[V]:
    """Store a value while preserving its exact type and identity."""

    def __init__(self, store: Callable[[V], None]) -> None:
        self._store = store

    # A bound such as ``W: V`` is not legal: TypeVar bounds cannot be generic.
    # Contravariant storage constrains W to accepted values without widening it
    # to V, preserving generic classes, overloads, and callable-object attributes.
    def __call__[W](self: _Registration[W], value: W) -> W:
        self._store(value)
        return value


class _FunctionRegistration[F]:
    """Identity-preserving registration with the function decorator's ``f`` keyword."""

    def __init__(self, store: Callable[[F], None]) -> None:
        self._store = store

    def __call__[W](self: _FunctionRegistration[W], f: W) -> W:
        self._store(f)
        return f


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
        self._registry: dict[object, Callable[Concatenate[T, In], Out]] = {}
        update_wrapper(self, func, updated=())

    @overload
    def register(self, key: T) -> _FunctionRegistration[Callable[Concatenate[T, In], Out]]: ...

    @overload
    def register[F](self: _RegistrationSink[T, F], key: T, f: F) -> F: ...

    def register(
        self,
        key: object,
        f: Callable[Concatenate[T, In], Out] | None = None,
    ) -> Callable[Concatenate[T, In], Out] | _FunctionRegistration[Callable[Concatenate[T, In], Out]]:
        """Register a function for the given key, returning it unchanged."""
        if f is None:
            return self.multi_register([key])
        return self.multi_register([key], f)

    @overload
    def multi_register(self, keys: Iterable[object]) -> _FunctionRegistration[Callable[Concatenate[T, In], Out]]: ...

    @overload
    def multi_register[F](self: _RegistrationSink[object, F], keys: Iterable[object], f: F) -> F: ...

    def multi_register(
        self,
        keys: Iterable[object],
        f: Callable[Concatenate[T, In], Out] | None = None,
    ) -> Callable[Concatenate[T, In], Out] | _FunctionRegistration[Callable[Concatenate[T, In], Out]]:
        """Register a function for the given keys, returning it unchanged."""
        if f is not None:
            for key in keys:
                self._register_value(key, f)
            return f

        def store(f: Callable[Concatenate[T, In], Out]) -> None:
            for key in keys:
                self._register_value(key, f)

        return _FunctionRegistration(store)

    def _register_value(self, key: object, value: Callable[Concatenate[T, In], Out]) -> None:
        self._registry[key] = value

    def __call__(self, arg: T, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Call the appropriate function based on the argument."""
        func_impl = self._registry.get(arg, self._func)
        return func_impl(arg, *args, **kwargs)


class switch[K, V](MutableMapping[K, V]):  # noqa: N801
    """A simple switch-case mechanism based on a dictionary."""

    _registry: dict[K, V]

    def __init__(
        self,
        cases: dict[K, V] | None = None,
    ) -> None:
        """Initialize the switch with the given cases and default value."""
        self._registry = cases or {}

    @overload
    def register(self, key: K) -> _Registration[V]: ...

    @overload
    def register[W](self: _RegistrationSink[K, W], key: K, value: W) -> W: ...

    def register(
        self,
        key: K,
        value: V | NotImplementedType = NotImplemented,
    ) -> _Registration[V] | V:
        """Register a value for the given key, returning it unchanged."""
        if value is NotImplemented:
            return self.multi_register([key])
        return self.multi_register([key], value)

    @overload
    def multi_register(self, keys: Iterable[K]) -> _Registration[V]: ...

    @overload
    def multi_register[W](self: _RegistrationSink[K, W], keys: Iterable[K], value: W) -> W: ...

    def multi_register(
        self,
        keys: Iterable[K],
        value: V | NotImplementedType = NotImplemented,
    ) -> _Registration[V] | V:
        """Register a value for the given keys, returning it unchanged."""
        if value is not NotImplemented:
            for key in keys:
                self._register_value(key, value)
            return value

        def store(value: V) -> None:
            for key in keys:
                self._register_value(key, value)

        return _Registration(store)

    def _register_value(self, key: K, value: V) -> None:
        self._registry[key] = value

    @override
    def __getitem__(self, key: K) -> V:
        """Return the value corresponding to the given key."""
        return self._registry[key]

    def __call__(self, key: K) -> V:
        """Return the value corresponding to the given key."""
        return self._registry[key]

    @override
    def __setitem__(self, key: K, value: V) -> None:
        """Set the value for the given key."""
        self._registry[key] = value

    @override
    def __delitem__(self, key: K) -> None:
        """Delete the value for the given key."""
        del self._registry[key]

    @override
    def __iter__(self) -> Iterator[K]:
        """Return an iterator over the keys in the switch cases."""
        return iter(self._registry)

    @override
    def __len__(self) -> int:
        """Return the number of cases in the switch."""
        return len(self._registry)
