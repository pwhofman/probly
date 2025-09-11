"""A lazy version of functools.singledispatch."""

from __future__ import annotations

from functools import singledispatch, update_wrapper
import types
from typing import TYPE_CHECKING, Any, Union, get_args

from lazy_dispatch.isinstance import (
    LazyType,
    _find_closest_string_type,
    _is_union_type,
    _split_lazy_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def is_valid_dispatch_type(cls: LazyType) -> bool:
    """Check if cls is a valid dispatch type."""
    if isinstance(cls, (type, str)):
        return True
    if isinstance(cls, tuple):
        return all(is_valid_dispatch_type(c) for c in cls)
    if _is_union_type(cls):
        return all(is_valid_dispatch_type(c) for c in get_args(cls))
    return False


def lazy_singledispatch(func: Callable) -> Callable:  # noqa: C901
    """A lazy version of functools.singledispatch that also works with string types."""
    singledispatcher = singledispatch(func)
    eager_dispath = singledispatcher.dispatch
    eager_register: Callable = singledispatcher.register
    funcname = getattr(func, "__name__", "singledispatch function")
    string_registry: dict[str, Callable] = {}

    def lazy_dispatch(cls: type) -> Callable:
        if len(string_registry) > 0:
            closest = _find_closest_string_type(cls, string_registry)
            if closest is not None:
                real_type, string_type = closest
                f = string_registry.pop(string_type)
                eager_register(real_type, f)

        return eager_dispath(cls)

    def lazy_register(cls: LazyType | Callable, func: Callable | None = None) -> Callable:
        nonlocal string_registry
        if is_valid_dispatch_type(cls):  # type: ignore[arg-type]
            if func is None:
                return lambda f: lazy_register(cls, f)
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
            func = cls  # type: ignore[assignment]

            argname, cls = next(iter(func.__annotations__.items()))
            if not is_valid_dispatch_type(cls):  # type: ignore[arg-type]
                if _is_union_type(cls) or isinstance(cls, tuple):
                    msg = f"Invalid annotation for {argname!r}. {cls!r} not all arguments are classes or strings."
                    raise TypeError(msg)
                msg = f"Invalid annotation for {argname!r}. {cls!r} is not a class or string."
                raise TypeError(msg)

        types, strings = _split_lazy_type(cls)  # type: ignore[arg-type]

        if len(types) > 0:
            eager_register(Union.__getitem__(tuple(types)), func)

        if len(strings) > 0:
            for s in strings:
                string_registry[s] = func  # type: ignore[assignment]

        return func  # type: ignore[return-value]

    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if not args:
            msg = f"{funcname} requires at least 1 positional argument"
            raise TypeError(msg)
        return lazy_dispatch(args[0].__class__)(*args, **kwargs)

    wrapper.register = lazy_register  # type: ignore[attr-defined]
    wrapper.dispatch = lazy_dispatch  # type: ignore[attr-defined]
    wrapper.string_registry = types.MappingProxyType(string_registry)  # type: ignore[attr-defined]
    update_wrapper(wrapper, func)

    return wrapper
