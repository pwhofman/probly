"""Torch G-VBLL implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from probly.layers.torch import GVBLLLayer

from ._common import (
    DOF,
    LAST_LAYER,
    NOISE_INIT,
    PRIOR_SCALE,
    WISHART_SCALE,
    g_vbll_traverser,
)

if TYPE_CHECKING:
    from pytraverse import State


@g_vbll_traverser.register(nn.Linear)
def replace_linear_with_g_vbll(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the last linear layer with a `GVBLLLayer`; leave the rest untouched."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
        return GVBLLLayer(
            in_features=obj.in_features,
            num_classes=obj.out_features,
            prior_scale=state[PRIOR_SCALE],
            noise_init=state[NOISE_INIT],
            wishart_scale=state[WISHART_SCALE],
            dof=state[DOF],
        ), state
    return obj, state


@g_vbll_traverser.register(nn.Softmax)
def remove_g_vbll_softmax(obj: nn.Softmax, state: State) -> tuple[nn.Module, State]:
    """Remove a trailing softmax layer; the G-VBLL layer outputs logits."""
    if state[LAST_LAYER]:
        return nn.Identity(), state
    return obj, state


@g_vbll_traverser.register(nn.Module)
def skip_other_g_vbll_modules(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Leave other modules unchanged."""
    return obj, state
