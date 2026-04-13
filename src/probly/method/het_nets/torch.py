"""Torch implementation of HetNets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from probly.layers.torch import HeteroscedasticLayer

from ._common import IS_PARAMETER_EFFICIENT, LAST_LAYER, MULTILABEL, NUM_FACTORS, NUM_MC_SAMPLES, TEMPERATURE, register

if TYPE_CHECKING:
    from pytraverse import State


def skip_layer(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Traverser for torch HetNets."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
    return obj, state


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
            num_mc_samples=state[NUM_MC_SAMPLES],
            is_parameter_efficient=state[IS_PARAMETER_EFFICIENT],
            multilabel=state[MULTILABEL],
        ), state
    return obj, state


def ignore_layer(obj: nn.Sequential, state: State) -> tuple[nn.Module, State]:
    """Skip last layer if it is a sequential."""
    return obj, state


register(nn.Module, skip_layer)
register(nn.Sequential, ignore_layer)
register(nn.Linear, drop_in_place_het_layer)
