"""Traversal implementation for flax modules."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from flax.nnx.helpers import Sequential
from flax.nnx.module import Module

import pytraverse as t
from pytraverse import generic
from pytraverse.decorators import traverser

from . import common as tnn

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# Torch traversal variables

ROOT = t.StackVariable[Module | None]("ROOT", "A reference to the outermost module.")
CLONE = t.StackVariable[bool](
    "CLONE",
    "Whether to clone torch modules before making changes.",
    default=generic.CLONE,
)
TRAVERSE_REVERSED = t.StackVariable[bool](
    "TRAVERSE_REVERSED",
    "Whether to traverse elements in reverse order.",
    default=generic.TRAVERSE_REVERSED,
)
FLATTEN_SEQUENTIAL = t.StackVariable[bool](
    "FLATTEN_SEQUENTIAL",
    "Whether to flatten sequential flax modules after making changes.",
    default=tnn.FLATTEN_SEQUENTIAL,
)

# Torch model cloning


@traverser(type=Module)
def _clone_traverser(
    obj: Module,
    state: t.State[Module],
) -> t.TraverserResult[Module]:
    if state[CLONE]:
        obj = copy.deepcopy(obj)
        # Do not clone the module twice:
        state[CLONE] = False
        # After deepcopy, generic datastructures will have been cloned as well:
        state[generic.CLONE] = False

    return obj, state


# Flax model root tracking


@traverser(type=Module)
def _root_traverser(
    obj: Module,
    state: t.State[Module],
) -> t.TraverserResult[Module]:
    if state[ROOT] is None:
        state[ROOT] = obj
        state[tnn.LAYER_COUNT] = 0
    return obj, state


# Flax model layer counting


@tnn.layer_count_traverser.register(vars={"count": tnn.LAYER_COUNT}, update_vars=True)
def _module_counter(obj: Module, count: int) -> tuple[Module, dict[str, int]]:
    return obj, {
        "count": count + 1,  # Increment LAYER_COUNT for each traversed module.
    }


@tnn.layer_count_traverser.register
def _sequential_counter(obj: Sequential) -> Sequential:
    return obj  # Don't count sequential modules as layers.


# Flax model traverser

_torch_traverser = t.singledispatch_traverser[Module](name="_torch_traverser")


@_torch_traverser.register
def _module_traverser(
    obj: Module,
    state: t.State[Module],
    traverse: t.TraverserCallback[Module],
) -> t.TraverserResult[Module]:
    children: Iterator[tuple[str, Module]] = obj.iter_children()  # type: ignore[assignment]
    if state[TRAVERSE_REVERSED]:
        children = reversed(list(children))
    for name, module in children:
        new_module, state = traverse(module, state, name)
        setattr(obj, name, new_module)

    return obj, state


@_torch_traverser.register
def _sequential_traverser(
    obj: Sequential,
    state: t.State[Module],
    traverse: t.TraverserCallback[Module],
) -> t.TraverserResult[Module]:
    if not state[FLATTEN_SEQUENTIAL]:
        return _module_traverser(obj, state, traverse)

    seq: list[Module] = []
    children: Iterable[Module] = obj.layers  # type: ignore[assignment]
    traverse_reversed = state[TRAVERSE_REVERSED]
    if traverse_reversed:
        children = reversed(list(children))

    for module in children:
        new_module, state = traverse(module, state)
        if isinstance(new_module, Sequential):
            sub_children: Iterable[Module] = new_module.layers  # type: ignore[assignment]
            if traverse_reversed:
                sub_children = reversed(list(sub_children))
            seq += sub_children
        else:
            seq.append(new_module)

    new_obj = Sequential(*(reversed(seq) if traverse_reversed else seq))

    return new_obj, state


# Public API combining cloning, root tracking, and module traversing

torch_traverser: t.Traverser[Module] = t.sequential(
    _clone_traverser,
    _root_traverser,
    _torch_traverser,
    name="torch_traverser",
)
torch_traverser.register = _torch_traverser.register  # type: ignore[attr-defined]

tnn.nn_traverser.register(Module, torch_traverser)
