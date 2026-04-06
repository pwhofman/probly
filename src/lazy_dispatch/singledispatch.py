"""A lazy version of functools.singledispatch."""

from __future__ import annotations

from functools import reduce, singledispatch, update_wrapper
import operator
from typing import TYPE_CHECKING, Any, get_args, overload

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


class Lazydispatch[**In, Out]:
    """A lazy version of functools.singledispatch that also works with string types."""

    __slots__ = (
        "__dict__",
        "__weakref__",
        "_parse_annotations",
        "_singledispatcher",
        "delayed_registration_registry",
        "dispatch_on",
        "funcname",
        "registry_meta_types",
        "string_registry",
    )

    def __init__(
        self,
        func: Callable[In, Out] | None = None,
        *,
        dispatch_on: Callable[In, Any] = first_argument,
        parse_annotations: bool = True,
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
        self.dispatch_on: Callable[In, Any] = dispatch_on
        self._parse_annotations = parse_annotations

    def dispatch(  # noqa: C901, PLR0912
        self,
        cls: type,
        *,
        delayed_register: bool = True,
        registry_meta_lookup: object = NotImplemented,
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
                if isinstance(real_type, RegistryMeta):
                    self.registry_meta_types.add(real_type)

        f = self._singledispatcher.dispatch(cls)

        if registry_meta_lookup is not NotImplemented:
            # Check if an instance-level registry registration applies.
            registry_meta_matches: set[type] = set()
            for registry_meta_type in self.registry_meta_types:
                if isinstance(registry_meta_lookup, registry_meta_type):
                    should_add = True
                    if len(registry_meta_matches) > 0:
                        for registry_meta_match in list(registry_meta_matches):
                            if issubclass(registry_meta_type, registry_meta_match):
                                registry_meta_matches.remove(registry_meta_match)
                                continue
                            if issubclass(registry_meta_match, registry_meta_type):
                                should_add = False
                                continue
                    if should_add:
                        registry_meta_matches.add(registry_meta_type)

            if len(registry_meta_matches) > 0:
                if len(registry_meta_matches) > 1:
                    msg = f"Ambiguous dispatch: {registry_meta_matches!r}."
                    raise RuntimeError(msg)

                registry_meta_match = next(iter(registry_meta_matches))

                non_parent_matches = []
                for registered_type in self._singledispatcher.registry:
                    registered_func = self._singledispatcher.registry[registered_type]
                    if registered_func is f:
                        if issubclass(registered_type, registry_meta_match):
                            return f
                        if not issubclass(registry_meta_match, registered_type):
                            non_parent_matches.append(registered_type)

                for non_parent_match in non_parent_matches:
                    if isinstance(registry_meta_lookup, non_parent_match):
                        msg = f"Ambiguous dispatch: {non_parent_match!r} or {registry_meta_match!r}."
                        raise RuntimeError(msg)  # noqa: TRY004

                return self._singledispatcher.dispatch(registry_meta_match)

        return f

    def eager_register(self, cls: type | UnionType | Callable, func: Callable | None = None) -> Callable:
        """Eagerly register a new implementation for the given type or union type.

        This method simply delegates to the underlying functools.singledispatch instance.
        Instance-based RegistryMeta types and lazy-string registrations are not supported.
        To use the full functionality of lazydispatch, use the `register` method instead.
        """
        return self._singledispatcher.register(cls, func)  # ty: ignore[no-matching-overload]

    def register(self, cls: LazyType | Callable, func: Callable | None = None) -> Callable:  # noqa: PLR0912
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

            if self._parse_annotations:
                # Only parse and dereference string annotations when not in lazy dispatch mode.
                from typing import get_type_hints  # noqa: PLC0415

                argname, cls = next(iter(get_type_hints(func).items()))
            else:
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

            if self._parse_annotations:
                # Only parse and dereference string annotations when not in lazy dispatch mode.
                from typing import get_type_hints  # noqa: PLC0415

                argname, cls = next(iter(get_type_hints(func).items()))
            else:
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
        dispatch_value = self.dispatch_on(*args, **kwargs)  # ty:ignore[invalid-argument-type]
        return self.dispatch(
            dispatch_value.__class__,
            registry_meta_lookup=dispatch_value,
        )(*args, **kwargs)


@overload
def lazydispatch[**In, Out](
    func: Callable[In, Out],
    *,
    dispatch_on: Callable = first_argument,
    parse_annotations: bool = True,
) -> Lazydispatch[In, Out]: ...


@overload
def lazydispatch[**In, Out](
    *,
    dispatch_on: Callable = first_argument,
    parse_annotations: bool = True,
) -> Callable[[Callable[In, Out]], Lazydispatch[In, Out]]: ...


def lazydispatch[**In, Out](
    func: Callable[In, Out] | None = None,
    *,
    dispatch_on: Callable = first_argument,
    parse_annotations: bool = True,
) -> Lazydispatch[In, Out] | Callable[[Callable[In, Out]], Lazydispatch[In, Out]]:
    """Create a new lazy_singledispatch or return a decorator."""
    if func is None:

        def decorator(func: Callable[In, Out]) -> Lazydispatch[In, Out]:
            return Lazydispatch(
                func,
                dispatch_on=dispatch_on,
                parse_annotations=parse_annotations,
            )

        return decorator

    return Lazydispatch(
        func,
        dispatch_on=dispatch_on,
        parse_annotations=parse_annotations,
    )
