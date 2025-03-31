"""
Traverser for standard Python datatypes (tuples, lists, dicts, sets).
"""

from typing import Any

from probly.traverse.composition import singledispatch_traverser
from probly.traverse.core import (
    StackVariable,
    State,
    TraverserCallback,
    TraverserResult,
)

CLONE = StackVariable[bool](
    "CLONE", "Whether to clone datastructures before making changes.", default=True
)
TRAVERSE_KEYS = StackVariable[bool](
    "TRAVERSE_KEYS", "Whether to traverse the keys of dictionaries.", default=False
)


generic_traverser = singledispatch_traverser(name="generic_traverser")


@generic_traverser.register
def _tuple_traverser(
    obj: tuple, state: State[tuple], traverse: TraverserCallback[Any]
) -> TraverserResult[tuple]:
    new_obj = []
    for i, o in enumerate(obj):
        o, state = traverse(o, state, i)
        new_obj.append(o)
    return tuple(new_obj), state


@generic_traverser.register
def _list_traverser(
    obj: list, state: State[list], traverse: TraverserCallback[Any]
) -> TraverserResult[list]:
    if state[CLONE]:
        new_obj = obj.__class__()
        for i, o in enumerate(obj):
            o, state = traverse(o, state, i)
            new_obj.append(o)
        return new_obj, state

    for i, o in enumerate(obj):
        o, state = traverse(o, state, i)
        obj[i] = o
    return obj, state


@generic_traverser.register
def _dict_traverser(
    obj: dict, state: State[dict], traverse: TraverserCallback[Any]
) -> TraverserResult[dict]:
    traverse_keys = state[TRAVERSE_KEYS]
    if state[CLONE] or traverse_keys:
        new_obj = obj.__class__()
        for k, v in obj.items():
            if traverse_keys:
                k, state = traverse(k, state)
            v, state = traverse(v, state, k)
            new_obj[k] = v
        return new_obj, state

    for k, v in obj.items():
        v, state = traverse(v, state, k)
        obj[k] = v

    return obj, state


@generic_traverser.register
def _set_traverser(
    obj: set, state: State[set], traverse: TraverserCallback[Any]
) -> TraverserResult[set]:
    new_obj = obj.__class__()
    for o in obj:
        o, state = traverse(o, state)
        new_obj.add(o)
    return new_obj, state
