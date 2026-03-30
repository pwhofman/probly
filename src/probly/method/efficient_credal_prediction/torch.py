"""Torch specific functions for efficient credal prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from .common import LAST_LAYER, register

if TYPE_CHECKING:
    from pytraverse import State


def skip_layer(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Traverser for torch efficient credal prediction."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
    return obj, state


def remove_softmax(obj: nn.Softmax, state: State) -> tuple[nn.Module, State]:
    """Remove the last layer if Softmax for the efficient credal prediction."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
        return nn.Sequential(), state
    return obj, state


def ignore_layer(obj: nn.Sequential, state: State) -> tuple[nn.Module, State]:
    """Skip last layer if it is a sequential."""
    return obj, state


register(nn.Module, skip_layer)
register(nn.Sequential, ignore_layer)
register(nn.Softmax, remove_softmax)
