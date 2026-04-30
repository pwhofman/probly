"""Torch implementation of heteroscedastic classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from probly.layers.torch import HeteroscedasticLayer

from ._common import (
    IS_PARAMETER_EFFICIENT,
    LAST_LAYER,
    NUM_FACTORS,
    TEMPERATURE,
    heteroscedastic_classification_traverser,
)

if TYPE_CHECKING:
    from pytraverse import State


@heteroscedastic_classification_traverser.register(nn.Module)
def skip_layer(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Traverser for unchanged torch layers."""
    return obj, state


@heteroscedastic_classification_traverser.register(nn.Linear)
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


@heteroscedastic_classification_traverser.register(nn.Softmax)
def remove_layer(obj: nn.Softmax, state: State) -> tuple[nn.Module, State]:
    """Remove the softmax layer."""
    if state[LAST_LAYER]:
        return nn.Identity(), state
    return obj, state
