import types
from collections.abc import Callable
from functools import singledispatch
from typing import Any, Protocol, Union, Unpack, get_origin, overload

from . import decorators as d
from .core import (
    State,
    Traverser,
    TraverserCallback,
    TraverserResult,
    identity_traverser,
)


def sequential[T](*traversers: Traverser[T], name: str | None = None) -> Traverser[T]:
    def _traverser(
        obj: T,
        state: State[T],
        traverse: TraverserCallback[T],
    ) -> TraverserResult[T]:

        for traverser in traversers:
            obj, state = traverser(obj, state, traverse)
        return obj, state

    if name is not None:
        _traverser.__name__ = name
        _traverser.__qualname__ = f"{__name__}.{name}"

    return _traverser


def top_sequential[T](
    *traversers: Traverser[T], name: str | None = None
) -> Traverser[T]:

    def _traverser(
        obj: T,
        state: State[T],
        traverse: TraverserCallback[T],
    ) -> TraverserResult[T]:
        for traverser in traversers:

            def fixed_next(
                obj: T,
                state: State[T],
                meta: Any = None,
                traverser: Traverser[T] | None = traverser,
            ) -> TraverserResult[T]:
                return traverse(obj, state, meta, traverser)

            obj, state = traverser(obj, state, fixed_next)

        return obj, state

    if name is not None:
        _traverser.__name__ = name
        _traverser.__qualname__ = f"{__name__}.{name}"

    return _traverser


def _is_union_type(cls):
    return get_origin(cls) in {Union, types.UnionType}


def _is_valid_dispatch_type(cls):
    if isinstance(cls, type):
        return True
    from typing import get_args

    return _is_union_type(cls) and all(isinstance(arg, type) for arg in get_args(cls))


class RegisteredLooseTraverser[T](Protocol):
    def __call__(self, obj: T, *args, **kwargs) -> Any: ...


class singledispatch_traverser[T]:
    """
    A wrapper around functools.singledispatch to create an extensible traverser.
    All registered traversers are automatically wrapped in a `@traverser` decorator.
    """

    __slots__ = (
        "__name__",
        "__qualname__",
        "_dispatch",
    )

    def __init__(
        self,
        traverser: RegisteredLooseTraverser | None = None,
        *,
        name: str | None = None,
    ):
        self._dispatch = singledispatch(identity_traverser)

        if traverser is not None:
            if name is None:
                if hasattr(traverser, "__name__"):
                    self.__name__ = traverser.__name__  # type: ignore
                self.__qualname__ = traverser.__qualname__
            self.register(traverser)

        if name is not None:
            self.__name__ = name
            self.__qualname__ = f"{__name__}.{name}"

    def __call__(
        self,
        obj: T,
        state: State[T],
        traverse: TraverserCallback[T],
    ) -> TraverserResult[T]:
        return self._dispatch(obj, state, traverse)

    @overload
    def register(
        self, **kwargs: Unpack[d.TraverserDecoratorKwargs]
    ) -> Callable[[RegisteredLooseTraverser[T]], Traverser[T]]: ...

    @overload
    def register(
        self, cls: type | types.UnionType, **kwargs: Unpack[d.TraverserDecoratorKwargs]
    ) -> Callable[[RegisteredLooseTraverser[T]], Traverser[T]]: ...

    @overload
    def register(
        self,
        cls: RegisteredLooseTraverser[T],
        **kwargs: Unpack[d.TraverserDecoratorKwargs],
    ) -> Traverser[T]: ...

    @overload
    def register(
        self,
        cls: Any,
        traverser: RegisteredLooseTraverser[T],
        **kwargs: Unpack[d.TraverserDecoratorKwargs],
    ) -> Traverser[T]: ...

    def register(  # type: ignore
        self,
        cls=None,
        traverser: RegisteredLooseTraverser[T] | None = None,
        **kwargs: Unpack[d.TraverserDecoratorKwargs],
    ):
        if cls is not None:
            if _is_valid_dispatch_type(cls):
                if traverser is None:

                    def partial_register(
                        traverser: RegisteredLooseTraverser[T],
                    ) -> Traverser[T]:
                        return self.register(cls, traverser, **kwargs)

                    return partial_register
            else:
                if traverser is not None:
                    raise TypeError(f"Invalid first argument to `register()`: {cls!r}.")
                traverser = cls
                cls = None
        else:
            if traverser is not None:
                raise TypeError(
                    f"Invalid arguments to `register(None, {traverser!r}.)`"
                )

            def partial_register(
                traverser: RegisteredLooseTraverser[T],
            ) -> Traverser[T]:
                return self.register(traverser, **kwargs)

            return partial_register

        traverser = d.traverser(traverser, **kwargs)  # type: ignore

        if cls is not None:
            return self._dispatch.register(cls, traverser)

        return self._dispatch.register(traverser)
