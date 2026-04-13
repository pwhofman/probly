"""Torch credal net implementation."""

from __future__ import annotations

from torch import nn

from probly.layers.torch import IntSoftmax
from pytraverse import GlobalVariable, State

from ._common import credal_net_traverser

REPLACED = GlobalVariable[bool]("REPLACED", default=False)


@credal_net_traverser.register(nn.Module)
def _(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Traverser for torch credal net."""
    if state[REPLACED]:
        return obj, state
    return nn.Sequential(), state


@credal_net_traverser.register(nn.Linear)
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the last linear layer with a new head for the credal net."""
    if state[REPLACED]:
        return obj, state
    state[REPLACED] = True
    # replace last linear layer with new head
    new_head = nn.Sequential(
        nn.Linear(obj.in_features, 2 * obj.out_features),
        nn.BatchNorm1d(2 * obj.out_features),
        IntSoftmax(),
    )
    return new_head, state
