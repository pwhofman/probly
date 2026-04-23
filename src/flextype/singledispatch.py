"""A lazy version of functools.singledispatch."""

from __future__ import annotations

from contextlib import suppress
from functools import reduce, singledispatch, update_wrapper
import operator
from typing import TYPE_CHECKING, Any, SupportsIndex, get_args, overload, override
from weakref import WeakKeyDictionary

from flextype.isinstance import (
    LazyType,
    _find_closest_string_type,
    _is_union_type,
    _split_lazy_type,
    lazy_issubclass,
)
from flextype.registry_meta import RegistryMeta

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _resolve_global(module_name: str, qualname: str) -> object:
    """Resolve a global object by module + qualname."""
    import importlib  # noqa: PLC0415

    obj: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        if part == "<locals>":
            msg = f"Can't pickle local object {module_name}.{qualname}"
            raise AttributeError(msg)
        obj = getattr(obj, part)
    return obj


class Flexdispatch[**In, Out]:
    """A lazy version of functools.singledispatch that also works with string types."""

    __slots__ = (
        "__dict__",
        "__weakref__",
        "_delayed_miss_cache",
        "_delayed_registry_generation",
        "_parse_annotations",
        "_singledispatcher",
        "_string_miss_cache",
        "_string_registry_generation",
        "_types_by_func",
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
        self._delayed_registry_generation = 0
        self._string_registry_generation = 0
        self._delayed_miss_cache: WeakKeyDictionary[type, int] = WeakKeyDictionary()
        self._string_miss_cache: WeakKeyDictionary[type, int] = WeakKeyDictionary()
        self._types_by_func: dict[Callable, set[type]] = {}
        for registered_type, registered_func in self._singledispatcher.registry.items():
            with suppress(TypeError):
                self._types_by_func.setdefault(registered_func, set()).add(registered_type)

    def _bump_delayed_registry_generation(self) -> None:
        self._delayed_registry_generation += 1
        self._delayed_miss_cache.clear()

    def _bump_string_registry_generation(self) -> None:
        self._string_registry_generation += 1
        self._string_miss_cache.clear()

    def dispatch(  # noqa: C901, PLR0912, PLR0915
        self,
        cls: type,
        *,
        delayed_register: bool = True,
        registry_meta_lookup: object = NotImplemented,
    ) -> Callable[..., Out]:
        """Find the best available function for the given type or string."""
        delayed_registration_registry = self.delayed_registration_registry
        string_registry = self.string_registry
        if delayed_register and delayed_registration_registry:
            delayed_generation = self._delayed_registry_generation
            delayed_miss_generation = self._delayed_miss_cache.get(cls)
            if delayed_miss_generation != delayed_generation:
                active_registrations: dict[str | type, RegistrationFunction] = {
                    t: registration_func
                    for t, registration_func in delayed_registration_registry.items()
                    if lazy_issubclass(cls, t)
                }

                if active_registrations:
                    for registration_func in active_registrations.values():
                        registration_func(cls)

                    for t in active_registrations:
                        delayed_registration_registry.pop(t, None)

                    self._bump_delayed_registry_generation()
                else:
                    self._delayed_miss_cache[cls] = delayed_generation

        if string_registry:
            string_generation = self._string_registry_generation
            string_miss_generation = self._string_miss_cache.get(cls)
            if string_miss_generation != string_generation:
                closest = _find_closest_string_type(cls, string_registry)
                if closest is not None:
                    real_type, string_type = closest
                    registration_func = string_registry.pop(string_type)
                    self._bump_string_registry_generation()
                    self._eager_register((real_type,), registration_func)
                else:
                    self._string_miss_cache[cls] = string_generation

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

                try:
                    registered_types = self._types_by_func.get(f)
                except TypeError:
                    registered_types = None

                if registered_types is None:
                    registry = self._singledispatcher.registry
                    registered_types = {t for t, registered_func in registry.items() if registered_func is f}

                first_non_parent_match: type | None = None
                for registered_type in registered_types:
                    if issubclass(registered_type, registry_meta_match):
                        return f
                    if (
                        first_non_parent_match is None
                        and not issubclass(registry_meta_match, registered_type)
                        and isinstance(registry_meta_lookup, registered_type)
                    ):
                        first_non_parent_match = registered_type

                if first_non_parent_match is not None:
                    msg = f"Ambiguous dispatch: {first_non_parent_match!r} or {registry_meta_match!r}."
                    raise RuntimeError(msg)

                return self._singledispatcher.dispatch(registry_meta_match)

        return f

    def _eager_register(self, types: tuple[type], func: Callable) -> Callable:
        """Eagerly register a new implementation for the given type or union type.

        This method simply delegates to the underlying functools.singledispatch instance.
        Instance-based RegistryMeta types and lazy-string registrations are not supported.
        To use the full functionality of flexdispatch, use the `register` method instead.
        """
        registry = self._singledispatcher.registry
        old_functions = {t: registry.get(t) for t in types}

        if len(types) == 1:
            res = self._singledispatcher.register(types[0], func)
        else:
            res = self._singledispatcher.register(reduce(operator.or_, types), func)

        for t in types:
            if isinstance(t, RegistryMeta):
                self.registry_meta_types.add(t)

            old_function = old_functions[t]
            if old_function is not None and old_function is not func:
                try:
                    old_bucket = self._types_by_func.get(old_function)
                except TypeError:
                    old_bucket = None

                if old_bucket is not None:
                    old_bucket.discard(t)
                    if not old_bucket:
                        with suppress(TypeError):
                            del self._types_by_func[old_function]

            with suppress(TypeError):
                self._types_by_func.setdefault(func, set()).add(t)

        return res

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
            self._eager_register(tuple(types), func)  # ty:ignore[invalid-argument-type]

        if len(strings) > 0:
            for s in strings:
                self.string_registry[s] = func  # ty: ignore[invalid-assignment]
            self._bump_string_registry_generation()

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

        if types or strings:
            self._bump_delayed_registry_generation()

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

    @override
    def __reduce_ex__(self, protocol: SupportsIndex | None = None) -> tuple[object, tuple[str, str]]:
        """Pickle by global reference, like regular functions."""
        import pickle  # noqa: PLC0415

        del protocol
        module_name = getattr(self, "__module__", None)
        qualname = getattr(self, "__qualname__", None)
        if not isinstance(module_name, str) or not isinstance(qualname, str):
            msg = f"Can't pickle {self!r}: missing module/qualname."
            raise pickle.PicklingError(msg)

        try:
            resolved = _resolve_global(module_name, qualname)
        except Exception as err:
            msg = f"Can't pickle {self!r}: it's not found as {module_name}.{qualname}"
            raise pickle.PicklingError(msg) from err
        if resolved is not self:
            msg = f"Can't pickle {self!r}: it's not the same object as {module_name}.{qualname}"
            raise pickle.PicklingError(msg)
        return (_resolve_global, (module_name, qualname))

    @override
    def __reduce__(self) -> tuple[object, tuple[str, str]]:
        """Pickle by global reference, like regular functions."""
        return self.__reduce_ex__()


@overload
def flexdispatch[**In, Out](
    func: Callable[In, Out],
    *,
    dispatch_on: Callable = first_argument,
    parse_annotations: bool = True,
) -> Flexdispatch[In, Out]: ...


@overload
def flexdispatch[**In, Out](
    *,
    dispatch_on: Callable = first_argument,
    parse_annotations: bool = True,
) -> Callable[[Callable[In, Out]], Flexdispatch[In, Out]]: ...


def flexdispatch[**In, Out](
    func: Callable[In, Out] | None = None,
    *,
    dispatch_on: Callable = first_argument,
    parse_annotations: bool = True,
) -> Flexdispatch[In, Out] | Callable[[Callable[In, Out]], Flexdispatch[In, Out]]:
    """Create a new lazy_singledispatch or return a decorator."""
    if func is None:

        def decorator(func: Callable[In, Out]) -> Flexdispatch[In, Out]:
            return Flexdispatch(
                func,
                dispatch_on=dispatch_on,
                parse_annotations=parse_annotations,
            )

        return decorator

    return Flexdispatch(
        func,
        dispatch_on=dispatch_on,
        parse_annotations=parse_annotations,
    )
