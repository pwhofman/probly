"""Traverser for standard Python datatypes (tuples, lists, dicts, sets).

This module provides a generic traverser that can handle common Python data structures
using single dispatch. It includes configurable behavior for cloning and traversing
dictionary keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pytraverse.composition import SingledispatchTraverser
from pytraverse.core import (
    StackVariable,
    State,
    TraverserCallback,
    TraverserResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

CLONE = StackVariable[bool](
    "CLONE",
    "Whether to clone datastructures before making changes.",
    default=True,
)
TRAVERSE_KEYS = StackVariable[bool](
    "TRAVERSE_KEYS",
    "Whether to traverse the keys of dictionaries.",
    default=False,
)
TRAVERSE_REVERSED = StackVariable[bool](
    "TRAVERSE_REVERSED",
    "Whether to traverse elements in reverse order.",
    default=False,
)


generic_traverser = SingledispatchTraverser[object](name="generic_traverser")


@generic_traverser.register
def _tuple_traverser(
    obj: tuple,
    state: State[tuple[Any]],
    traverse: TraverserCallback[Any],
) -> TraverserResult[tuple[Any]]:
    """Traverse tuple elements and reconstruct the tuple.

    Always creates a new tuple since tuples are immutable.

    Args:
        obj: The tuple to traverse.
        state: Current traversal state.
        traverse: Callback for traversing child elements.

    Returns:
        A new tuple with traversed elements and updated state.
    """
    new_obj = []
    items: Iterable[tuple[int, Any]] = enumerate(obj)
    traverse_reversed = state[TRAVERSE_REVERSED]
    if traverse_reversed:
        items = reversed(list(items))
    for i, element in items:
        new_element, state = traverse(element, state, i)
        new_obj.append(new_element)
    if traverse_reversed:
        return tuple(reversed(new_obj)), state
    return tuple(new_obj), state


@generic_traverser.register
def _list_traverser(
    obj: list,
    state: State[list[Any]],
    traverse: TraverserCallback[Any],
) -> TraverserResult[list[Any]]:
    """Traverse list elements, optionally cloning the list.

    Behavior depends on the CLONE variable:
    - If True: Creates a new list with traversed elements
    - If False: Modifies the original list in-place

    Args:
        obj: The list to traverse.
        state: Current traversal state.
        traverse: Callback for traversing child elements.

    Returns:
        The modified or new list and updated state.
    """
    items: Iterable[tuple[int, Any]] = enumerate(obj)
    traverse_reversed = state[TRAVERSE_REVERSED]
    if traverse_reversed:
        items = reversed(list(items))

    if state[CLONE]:
        new_obj = obj.__class__()
        for i, element in items:
            new_element, state = traverse(element, state, i)
            new_obj.append(new_element)
        if traverse_reversed:
            new_obj.reverse()
        return new_obj, state

    for i, element in items:
        new_element, state = traverse(element, state, i)
        obj[i] = new_element
    return obj, state


@generic_traverser.register
def _dict_traverser(
    obj: dict,
    state: State[dict[Any, Any]],
    traverse: TraverserCallback[Any],
) -> TraverserResult[dict[Any, Any]]:
    """Traverse dictionary values and optionally keys.

    Behavior depends on CLONE and TRAVERSE_KEYS variables:
    - CLONE=True or TRAVERSE_KEYS=True: Creates a new dictionary
    - CLONE=False and TRAVERSE_KEYS=False: Modifies original dictionary
    - TRAVERSE_KEYS=True: Also traverses dictionary keys

    Args:
        obj: The dictionary to traverse.
        state: Current traversal state.
        traverse: Callback for traversing child elements.

    Returns:
        The modified or new dictionary and updated state.
    """
    traverse_keys = state[TRAVERSE_KEYS]
    items: Iterable[tuple[Any, Any]] = obj.items()
    traverse_reversed = state[TRAVERSE_REVERSED]
    if traverse_reversed:
        items = reversed(list(items))

    if state[CLONE] or traverse_keys:
        new_obj = obj.__class__()
        if traverse_reversed:
            additions = []
        for key, value in items:
            if traverse_keys:
                new_key, state = traverse(key, state)
            else:
                new_key = key
            new_value, state = traverse(value, state, new_key)
            if traverse_reversed:
                additions.append((new_key, new_value))
            else:
                new_obj[new_key] = new_value
        if traverse_reversed:
            new_obj.update(reversed(additions))
        return new_obj, state

    for key, value in items:
        new_value, state = traverse(value, state, key)
        obj[key] = new_value

    return obj, state


@generic_traverser.register
def _set_traverser(
    obj: set,
    state: State[set[Any]],
    traverse: TraverserCallback[Any],
) -> TraverserResult[set[Any]]:
    """Traverse set elements and reconstruct the set.

    Always creates a new set since sets are unordered and elements
    may change during traversal.

    Args:
        obj: The set to traverse.
        state: Current traversal state.
        traverse: Callback for traversing child elements.

    Returns:
        A new set with traversed elements and updated state.
    """
    new_obj = obj.__class__()
    for element in obj:
        new_element, state = traverse(element, state)
        new_obj.add(new_element)
    return new_obj, state
