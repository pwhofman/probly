"""Composition utilities for combining and creating complex traversers.

This module provides tools for composing multiple traversers into more complex
traversal behaviors. It includes sequential composition, top-level composition,
and single-dispatch traverser creation.
"""

from __future__ import annotations

from functools import singledispatch
import types
from typing import TYPE_CHECKING, Any, Protocol, Union, Unpack, get_origin, overload

from . import decorators as d
from .core import (
    State,
    Traverser,
    TraverserCallback,
    TraverserResult,
    identity_traverser,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def sequential[T](*traversers: Traverser[T], name: str | None = None) -> Traverser[T]:
    """Compose multiple traversers into a sequential execution chain.

    Creates a new traverser that applies the given traversers in sequence,
    where each traverser receives the output of the previous one as input.

    Args:
        *traversers: Variable number of traverser functions to compose in sequence.
        name: Optional name for the resulting traverser function.

    Returns:
        A new traverser that applies all input traversers sequentially.

    Example:
        >>> def add_one(obj, state, traverse):
        ...     return obj + 1, state
        >>> def multiply_two(obj, state, traverse):
        ...     return obj * 2, state
        >>> composed = sequential(add_one, multiply_two)
        >>> result = traverse(5, composed)  # ((5 + 1) * 2) = 12
    """

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
    *traversers: Traverser[T],
    name: str | None = None,
) -> Traverser[T]:
    """Compose multiple traversers with fixed next-traverser semantics.

    Creates a traverser that applies the given traversers in sequence, where each
    traverser receives a modified traverse callback that fixes the next traverser
    in the chain. This is useful for creating pipelines where each stage controls
    how the next stage is invoked.

    Args:
        *traversers: Variable number of traverser functions to compose.
        name: Optional name for the resulting traverser function.

    Returns:
        A new traverser that applies all input traversers with fixed next semantics.

    Example:
        >>> def preprocessor(obj, state, traverse):
        ...     # Do preprocessing
        ...     return traverse(obj, state)
        >>> def processor(obj, state, traverse):
        ...     # Do main processing
        ...     return traverse(obj, state)
        >>> composed = top_sequential(preprocessor, processor)
    """

    def _traverser(
        obj: T,
        state: State[T],
        traverse: TraverserCallback[T],
    ) -> TraverserResult[T]:
        for traverser in traversers:

            def fixed_next(
                obj: T,
                state: State[T],
                meta: Any = None,  # noqa: ANN401
                traverser: Traverser[T] | None = traverser,
            ) -> TraverserResult[T]:
                return traverse(obj, state, meta, traverser)

            obj, state = traverser(obj, state, fixed_next)

        return obj, state

    if name is not None:
        _traverser.__name__ = name
        _traverser.__qualname__ = f"{__name__}.{name}"

    return _traverser


def _is_union_type(cls: Any) -> bool:  # noqa: ANN401
    """Check if a type is a Union type.

    Args:
        cls: The type to check.

    Returns:
        True if the type is a Union type, False otherwise.
    """
    return get_origin(cls) in {Union, types.UnionType}


def _is_valid_dispatch_type(cls: Any) -> bool:  # noqa: ANN401
    """Check if a type is valid for single dispatch registration.

    Args:
        cls: The type to check.

    Returns:
        True if the type can be used for dispatch registration, False otherwise.
    """
    if isinstance(cls, type):
        return True
    from typing import get_args

    return _is_union_type(cls) and all(isinstance(arg, type) for arg in get_args(cls))


class RegisteredLooseTraverser[T](Protocol):
    """Protocol for traverser functions that can be registered with singledispatch.

    This protocol defines the interface for traverser functions before they are
    wrapped with the @traverser decorator.
    """

    def __call__(
        self,
        obj: T,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> TraverserResult[T] | T:
        """Call the traverser with flexible arguments.

        Args:
            obj: The object to traverse.
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            The result of traversal processing.
        """
        ...


class SingledispatchTraverser[T]:
    """A wrapper around functools.singledispatch to create an extensible traverser.

    This class provides a type-based dispatch mechanism for traversers, allowing
    different traversal logic to be registered for different object types. All
    registered traversers are automatically wrapped with the @traverser decorator.

    Type Parameters:
        T: The base type of objects that this traverser can handle.

    Example:
        >>> traverser = SingledispatchTraverser()
        >>> @traverser.register
        ... def _(obj: list, state, traverse):
        ...     return [traverse(item, state)[0] for item in obj], state
        >>> @traverser.register
        ... def _(obj: dict, state, traverse):
        ...     return {k: traverse(v, state)[0] for k, v in obj.items()}, state
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
    ) -> None:
        """Initialize a new SingledispatchTraverser.

        Args:
            traverser: Optional default traverser function to register.
            name: Optional name for the traverser.
        """
        self._dispatch = singledispatch(identity_traverser)

        if traverser is not None:
            if name is None:
                if hasattr(traverser, "__name__"):
                    self.__name__ = traverser.__name__  # type: ignore  # noqa: PGH003
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
        """Execute the appropriate traverser based on the object type.

        Args:
            obj: The object to traverse.
            state: The current traversal state.
            traverse: The traverser callback function.

        Returns:
            The result of the type-specific traversal.
        """
        return self._dispatch(obj, state, traverse)

    @overload
    def register(
        self,
        **kwargs: Unpack[d.TraverserDecoratorKwargs],
    ) -> Callable[[RegisteredLooseTraverser[T]], Traverser[T]]: ...

    @overload
    def register(
        self,
        cls: type | types.UnionType,
        **kwargs: Unpack[d.TraverserDecoratorKwargs],
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
        cls: Any,  # noqa: ANN401
        traverser: RegisteredLooseTraverser[T],
        **kwargs: Unpack[d.TraverserDecoratorKwargs],
    ) -> Traverser[T]: ...

    def register(  # type: ignore  # noqa: PGH003
        self,
        cls=None,
        traverser: RegisteredLooseTraverser[T] | None = None,
        **kwargs: Unpack[d.TraverserDecoratorKwargs],
    ):
        """Register a traverser for a specific type or as the default.

        This method supports multiple calling patterns:
        - @traverser.register: Register as default
        - @traverser.register(type): Register for specific type
        - traverser.register(type, function): Register function for type

        Args:
            cls: The type to register for, or the traverser function if no type.
            traverser: The traverser function when cls is a type.
            **kwargs: Additional arguments passed to the @traverser decorator.

        Returns:
            Either a decorator function or the registered traverser.

        Raises:
            TypeError: If invalid arguments are provided.
        """
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
                    msg = f"Invalid first argument to `register({cls!r})`."
                    raise TypeError(msg)
                traverser = cls
                cls = None
        else:
            if traverser is not None:
                msg = f"Invalid arguments to `register(None, {traverser!r})`."
                raise TypeError(msg)

            def partial_register(
                traverser: RegisteredLooseTraverser[T],
            ) -> Traverser[T]:
                return self.register(traverser, **kwargs)

            return partial_register

        traverser = d.traverser(traverser, **kwargs)  # type: ignore  # noqa: PGH003

        if cls is not None:
            return self._dispatch.register(cls, traverser)

        return self._dispatch.register(traverser)
