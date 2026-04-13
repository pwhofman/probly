"""Torch implementation of Credal Relative Likelihood."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ._common import BIAS_CLS, INITIALIZED, TOBIAS_VALUE, credal_relative_likelihood_traverser

if TYPE_CHECKING:
    from pytraverse.core import State


@credal_relative_likelihood_traverser.register(nn.Module)
def _(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Skip layers if last linear layer is initialized or raise error."""
    if not state[INITIALIZED]:
        msg = (
            f"Initialization of credal relative likelihood models "
            f"with last layer not being a linear layer is not possible. "
            f"Found last layer to be of type {type(obj)}"
        )
        raise ValueError(msg)
    return obj, state


@credal_relative_likelihood_traverser.register(nn.Linear)
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Initialize last linear layer with class bias."""
    if not state[INITIALIZED]:
        obj.bias.data[(state[BIAS_CLS] - 1) % obj.out_features] = state[TOBIAS_VALUE]
        state[INITIALIZED] = True
    return obj, state


@credal_relative_likelihood_traverser.register(nn.Softmax | nn.Sequential)
def _(obj: nn.Softmax | nn.Sequential, state: State) -> tuple[nn.Module, State]:
    """Skip softmax sequential layer at the end."""
    return obj, state
