"""Traverser for function and method types.

This module provides a traverser for wrapped functions and (bound) methods.

The main purpose of this traverser is to trigger traversal side effects on wrapped callables
and bound `self` objects.
"""

from __future__ import annotations

from types import FunctionType, MethodType
from typing import Any

from pytraverse.composition import SingledispatchTraverser
from pytraverse.core import (
    StackVariable,
    State,
    TraverserCallback,
    TraverserResult,
)

from . import generic

CLONE = StackVariable[bool](
    "CLONE",
    "Whether to clone functions before making changes.",
    default=generic.CLONE,
)
MODIFY_WRAPPERS = StackVariable[bool](
    "MODIFY_WRAPPERS",
    "Whether to allow modifications to function wrappers.",
    default=False,
)

function_traverser = SingledispatchTraverser[object](name="function_traverser")


@function_traverser.register
def _wrapped_function_traverser(
    obj: FunctionType,
    state: State[FunctionType],
    traverse: TraverserCallback[FunctionType],
) -> TraverserResult[FunctionType]:
    """Traverse function objects.

    Functions are treated as atomic and returned unchanged.

    Args:
        obj: The function to traverse.
        state: Current traversal state.
        traverse: Callback for traversing child elements.

    Returns:
        The original function and updated state.
    """
    if hasattr(obj, "__wrapped__"):
        new_wrapped, state = traverse(obj.__wrapped__, state, "__wrapped__")
        if new_wrapped is not obj.__wrapped__:
            if state[CLONE]:
                msg = "Cloning of wrapped functions is not supported."
                raise NotImplementedError(msg)
            if not state[MODIFY_WRAPPERS]:
                msg = "Modifying function wrappers is disabled."
                raise RuntimeError(msg)
            obj.__wrapped__ = new_wrapped

    return obj, state


@function_traverser.register
def _method_traverser(
    obj: MethodType,
    state: State[Any],
    traverse: TraverserCallback[Any],
) -> TraverserResult[MethodType]:
    """Traverse method objects.

    Allows traversing the underlying function and the instance the method is bound to.

    Args:
        obj: The method to traverse.
        state: Current traversal state.
        traverse: Callback for traversing child elements.

    Returns:
        The updated method and updated state.
    """
    func = obj.__func__
    func, state = traverse(func, state, "__func__")
    self = obj.__self__
    self, state = traverse(self, state, "__self__")
    if func is not obj.__func__ or self is not obj.__self__:
        return MethodType(func, self), state
    return obj, state
