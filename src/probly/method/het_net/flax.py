"""Flax implementation of HetNets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

from probly.layers.flax import HeteroscedasticLayer

from ._common import (
    IS_PARAMETER_EFFICIENT,
    LAST_LAYER,
    NUM_FACTORS,
    RNGS,
    TEMPERATURE,
    het_net_traverser,
)

if TYPE_CHECKING:
    from pytraverse import State


@het_net_traverser.register(nnx.Module)
def skip_layer(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    """Traverser for unchanged flax layers."""
    return obj, state


@het_net_traverser.register(nnx.Linear)
def drop_in_place_het_layer(obj: nnx.Linear, state: State) -> tuple[nnx.Module, State]:
    """Replace the last linear layer with a HeteroscedasticLayer."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
        return HeteroscedasticLayer(
            in_features=obj.in_features,
            num_classes=obj.out_features,
            num_factors=state[NUM_FACTORS],
            temperature=state[TEMPERATURE],
            is_parameter_efficient=state[IS_PARAMETER_EFFICIENT],
            rngs=state[RNGS],
        ), state
    return obj, state
