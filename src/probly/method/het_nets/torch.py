"""Torch implementation of HetNets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from probly.layers.torch import HeteroscedasticLayer

from ._common import (
    IS_PARAMETER_EFFICIENT,
    LAST_LAYER,
    NUM_FACTORS,
    TEMPERATURE,
    het_nets_traverser,
)

if TYPE_CHECKING:
    from pytraverse import State


@het_nets_traverser.register(nn.Module)
def skip_layer(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Traverser for torch HetNets."""
    return obj, state


@het_nets_traverser.register(nn.Linear)
def drop_in_place_het_layer(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the last linear layer with a HeteroscedasticLayer."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
        in_features = obj.in_features
        num_classes = obj.out_features
        return HeteroscedasticLayer(
            in_features=in_features,
            num_classes=num_classes,
            num_factors=state[NUM_FACTORS],
            temperature=state[TEMPERATURE],
            is_parameter_efficient=state[IS_PARAMETER_EFFICIENT],
        ), state
    return obj, state


@het_nets_traverser.register(nn.Softmax)
def remove_layer(obj: nn.Softmax, state: State) -> tuple[nn.Module, State]:
    """Remove the softmax layer."""
    if state[LAST_LAYER]:
        return nn.Identity(), state
    return obj, state
