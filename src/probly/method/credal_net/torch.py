"""Torch credal net implementation."""

from __future__ import annotations

from torch import nn

from probly.layers.torch import IntSoftmax
from probly.traverse_nn import nn_compose
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, State, singledispatch_traverser, traverse

from ._common import credal_net_generator

REPLACED = GlobalVariable[bool]("REPLACED", default=False)


@singledispatch_traverser
def torch_credal_net_traverser(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Traverser for torch credal net."""
    if state[REPLACED]:
        return obj, state
    return nn.Sequential(), state


@torch_credal_net_traverser.register
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


@credal_net_generator.register(cls=nn.Module)
def generate_torch_credal_net(model: nn.Module) -> nn.Module:
    """Build a torch credal net based on :cite:`wang2024credalnet`.

    Args:
        model: The torch model to be transformed.
    """
    model = traverse(model, nn_compose(torch_credal_net_traverser), init={TRAVERSE_REVERSED: True})
    return model
