"""A lazy version of functools.singledispatch."""

from __future__ import annotations

from functools import reduce, singledispatch, update_wrapper
import operator
from typing import TYPE_CHECKING, Any, Concatenate, get_args, overload

from lazy_dispatch.isinstance import (
    LazyType,
    _find_closest_string_type,
    _is_union_type,
    _split_lazy_type,
    lazy_issubclass,
)
from lazy_dispatch.registry_meta import RegistryMeta

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import UnionType

type RegistrationFunction = Callable[[type], Any]


def is_valid_dispatch_type(cls: LazyType) -> bool:
    """Check if cls is a valid dispatch type."""
    if isinstance(cls, (type, str)):
        return True
    if isinstance(cls, tuple):
        return all(is_valid_dispatch_type(c) for c in cls)
    if _is_union_type(cls):
        return all(is_valid_dispatch_type(c) for c in get_args(cls))
    return False


def first_argument[T](x: T, *args: Any, **kwargs: Any) -> T:  # noqa: ANN401, ARG001
    """Return the first argument."""
    return x


class Lazydispatch[T, **In, Out]:
    """A lazy version of functools.singledispatch that also works with string types."""

    def __init__(
        self,
        func: Callable[Concatenate[T, In], Out] | None = None,
        *,
        dispatch_on: Callable[Concatenate[T, In], Any] = first_argument,
    ) -> None:
        """Initialize the lazy_singledispatch instance."""
        if func is None:
            msg = "func must be provided"
            raise ValueError(msg)
        update_wrapper(self, func, updated=())

        self._singledispatcher = singledispatch(func)
        self.funcname = getattr(func, "__name__", "singledispatch function")
        self.string_registry: dict[str, Callable] = {}
        self.delayed_registration_registry: dict[str | type, RegistrationFunction] = {}
        self.registry_meta_types: set[RegistryMeta] = set()
        self.dispatch_on = dispatch_on

    def dispatch(
        self,
        cls: type,
        *,
        delayed_register: bool = True,
        registry_meta_lookup: object = None,
    ) -> Callable[..., Out]:
        """Find the best available function for the given type or string."""
        delayed_registration_registry = self.delayed_registration_registry
        string_registry = self.string_registry
        if delayed_register and len(delayed_registration_registry) > 0:
            active_registrations: dict[str | type, RegistrationFunction] = {
                t: registration_func
                for t, registration_func in delayed_registration_registry.items()
                if lazy_issubclass(cls, t)
            }

            for t, registration_func in active_registrations.items():
                registration_func(cls)
                del delayed_registration_registry[t]

        if len(string_registry) > 0:
            closest = _find_closest_string_type(cls, self.string_registry)
            if closest is not None:
                real_type, string_type = closest
                registration_func = string_registry.pop(string_type)
                self.eager_register(real_type, registration_func)

        f = self._singledispatcher.dispatch(cls)

        if registry_meta_lookup is not None and f is self._singledispatcher.registry[object]:
            registry_meta_match: RegistryMeta | None = None
            for registry_meta_type in self.registry_meta_types:
                if isinstance(registry_meta_lookup, registry_meta_type):
                    if registry_meta_match is not None:
                        if issubclass(registry_meta_type, registry_meta_match):
                            registry_meta_match = registry_meta_type
                            continue
                        if issubclass(registry_meta_match, registry_meta_type):
                            continue
                        msg = f"Ambiguous dispatch: {registry_meta_match!r} or {registry_meta_type!r}."
                        raise RuntimeError(msg)
                    registry_meta_match = registry_meta_type

            if registry_meta_match is not None:
                return self._singledispatcher.dispatch(registry_meta_match)

        return f

    def eager_register(self, cls: type | UnionType | Callable, func: Callable | None = None) -> Callable:
        """Eagerly register a new implementation for the given type or union type."""
        return self._singledispatcher.register(cls, func)  # ty: ignore[no-matching-overload]

    def register(self, cls: LazyType | Callable, func: Callable | None = None) -> Callable:
        """Register a new implementation for the given type or string."""
        if is_valid_dispatch_type(cls):  # ty: ignore[invalid-argument-type]
            if func is None:
                return lambda f: self.register(cls, f)
        else:
            if func is not None:
                msg = f"Invalid first argument to `register()`. {cls!r} is not a class, string, tuple or union type."
                raise TypeError(msg)
            ann = getattr(cls, "__annotations__", {})
            if not ann:
                msg = (
                    f"Invalid first argument to `register()`: {cls!r}. "
                    f"Use either `@register(some_class)` or plain `@register` "
                    f"on an annotated function."
                )
                raise TypeError(msg)
            func = cls  # ty: ignore[invalid-assignment]

            argname, cls = next(iter(func.__annotations__.items()))
            if not is_valid_dispatch_type(cls):
                if _is_union_type(cls) or isinstance(cls, tuple):
                    msg = f"Invalid annotation for {argname!r}. {cls!r} not all arguments are classes or strings."
                    raise TypeError(msg)
                msg = f"Invalid annotation for {argname!r}. {cls!r} is not a class or string."
                raise TypeError(msg)

        types, strings = _split_lazy_type(cls)  # ty: ignore[invalid-argument-type]

        if len(types) > 0:
            # Use reduce with operator.or_ to dynamically create a Union (PEP 604 style) and avoid private API usage.
            union_type = reduce(operator.or_, types)

            self.eager_register(union_type, func)

            for t in types:
                if isinstance(t, RegistryMeta):
                    self.registry_meta_types.add(t)

        if len(strings) > 0:
            for s in strings:
                self.string_registry[s] = func  # ty: ignore[invalid-assignment]

        return func  # ty: ignore[invalid-return-type]

    @overload
    def delayed_register(self, cls: LazyType) -> Callable[[RegistrationFunction], RegistrationFunction]: ...

    @overload
    def delayed_register(self, cls: RegistrationFunction) -> RegistrationFunction: ...

    @overload
    def delayed_register(self, cls: LazyType, func: RegistrationFunction) -> RegistrationFunction: ...

    def delayed_register(
        self,
        cls: LazyType | RegistrationFunction,
        func: RegistrationFunction | None = None,
    ) -> RegistrationFunction | Callable[[RegistrationFunction], RegistrationFunction]:
        """Register a delayed registration function."""
        if is_valid_dispatch_type(cls):  # ty: ignore[invalid-argument-type]
            if func is None:
                return lambda f: self.delayed_register(cls, f)  # ty: ignore[invalid-argument-type]
        else:
            if func is not None:
                msg = (
                    f"Invalid first argument to `delayed_register()`. "
                    f"{cls!r} is not a class, string, tuple or union type."
                )
                raise TypeError(msg)
            ann = getattr(cls, "__annotations__", {})
            if not ann:
                msg = (
                    f"Invalid first argument to `delayed_register()`: {cls!r}. "
                    f"Use either `@delayed_register(some_class)` or plain `@delayed_register` "
                    f"on an annotated function."
                )
                raise TypeError(msg)
            func = cls  # ty: ignore[invalid-assignment]

            argname, cls = next(iter(func.__annotations__.items()))
            if not is_valid_dispatch_type(cls):
                if _is_union_type(cls) or isinstance(cls, tuple):
                    msg = f"Invalid annotation for {argname!r}. {cls!r} not all arguments are classes or strings."
                    raise TypeError(msg)
                msg = f"Invalid annotation for {argname!r}. {cls!r} is not a class or string."
                raise TypeError(msg)

        types, strings = _split_lazy_type(cls)  # ty: ignore[invalid-argument-type]

        for t in types:
            self.delayed_registration_registry[t] = func  # ty: ignore[invalid-assignment]

        for s in strings:
            self.delayed_registration_registry[s] = func  # ty: ignore[invalid-assignment]

        return func  # ty: ignore[invalid-return-type]

    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Call the appropriate registered function based on the type of the first argument."""
        if not args:
            msg = f"{self.funcname} requires at least 1 positional argument"
            raise TypeError(msg)
        dispatch_value = self.dispatch_on(*args, **kwargs)
        return self.dispatch(
            dispatch_value.__class__,
            registry_meta_lookup=dispatch_value,
        )(*args, **kwargs)


@overload
def lazydispatch[T, **In, Out](
    func: Callable[Concatenate[T, In], Out], *, dispatch_on: Callable = first_argument
) -> Lazydispatch[T, In, Out]: ...


@overload
def lazydispatch[T, **In, Out](
    *, dispatch_on: Callable = first_argument
) -> Callable[[Callable[Concatenate[T, In], Out]], Lazydispatch[T, In, Out]]: ...


def lazydispatch[T, **In, Out](
    func: Callable[Concatenate[T, In], Out] | None = None,
    *,
    dispatch_on: Callable = first_argument,
) -> Lazydispatch[T, In, Out] | Callable[[Callable[Concatenate[T, In], Out]], Lazydispatch[T, In, Out]]:
    """Create a new lazy_singledispatch or return a decorator."""
    if func is None:

        def decorator(func: Callable[Concatenate[T, In], Out]) -> Lazydispatch[T, In, Out]:
            return Lazydispatch(func, dispatch_on=dispatch_on)

        return decorator

    return Lazydispatch(func, dispatch_on=dispatch_on)
