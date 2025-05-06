from __future__ import annotations

from collections import OrderedDict
import copy

from torch.nn import Module, Sequential

import probly.traverse as t
from probly.traverse import generic
from probly.traverse.decorators import traverser

from . import nn as tnn

ROOT = t.StackVariable[Module | None]("ROOT", "A reference to the outermost module.")
CLONE = t.StackVariable[bool](
    "CLONE",
    "Whether to clone torch modules before making changes.",
    default=generic.CLONE,
)
FLATTEN_SEQUENTIAL = t.StackVariable[bool](
    "FLATTEN_SEQUENTIAL",
    "Whether to flatten sequential modules after making changes.",
    default=True,
)


@traverser
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


@traverser
def _root_traverser(
    obj: Module,
    state: t.State[Module],
) -> t.TraverserResult[Module]:
    if state[ROOT] is None:
        state[ROOT] = obj
        state[tnn.LAYER_COUNT] = 0
    return obj, state


_torch_traverser = t.singledispatch_traverser(name="_torch_traverser")
torch_traverser = t.sequential(
    _clone_traverser, _root_traverser, _torch_traverser, name="torch_traverser"
)
tnn.nn_traverser.register(Module, torch_traverser)


@tnn.layer_count_traverser.register(vars={"count": tnn.LAYER_COUNT}, update_vars=True)
def _(obj: Module, count: int):
    return obj, {"count": count + 1}


@tnn.layer_count_traverser.register
@t.traverser
def _(obj: Sequential):
    return obj  # Don't count sequential modules as layers.


@_torch_traverser.register
def _module_traverser(
    obj: Module,
    state: t.State[Module],
    traverse: t.TraverserCallback[Module],
) -> t.TraverserResult[Module]:
    for name, module in obj.named_children():
        module, state = traverse(module, state, name)
        setattr(obj, name, module)

    return obj, state


@_torch_traverser.register
def _sequential_traverser(
    obj: Sequential, state: t.State[Module], traverse: t.TraverserCallback[Module]
) -> t.TraverserResult[Module]:
    if not state[FLATTEN_SEQUENTIAL]:
        return _module_traverser(obj, state, traverse)

    seq = []

    for name, module in obj.named_children():
        module, state = traverse(module, state, name)
        if isinstance(module, Sequential):
            for sub_name, sub_module in module.named_children():
                seq.append((f"{name}_{sub_name}", sub_module))
        else:
            seq.append((name, module))

    new_obj = Sequential(OrderedDict(seq))

    return new_obj, state
