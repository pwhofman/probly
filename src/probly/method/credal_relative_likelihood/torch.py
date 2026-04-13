"""Torch implementation of Credal Relative Likelihood."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ._common import BIAS_CLS, INITIALIZED, TOBIAS_VALUE, credal_relative_likelihood_traverser

if TYPE_CHECKING:
    from pytraverse.core import State


@credal_relative_likelihood_traverser.register(nn.Module)
def _(base: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Skip layers if last linear layer is initialized or raise error."""
    if not state[INITIALIZED]:
        msg = (
            f"Initialization of credal relative likelihood models "
            f"with last layer not being a linear layer is not possible. "
            f"Found last layer to be of type {type(base)}"
        )
        raise ValueError(msg)
    return base, state


@credal_relative_likelihood_traverser.register(nn.Linear)
def _(base: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Initialize last linear layer with class bias."""
    if not state[INITIALIZED]:
        base.bias.data[(state[BIAS_CLS] - 1) % base.out_features] = state[TOBIAS_VALUE]
        state[INITIALIZED] = True
    return base, state


@credal_relative_likelihood_traverser.register(nn.Softmax)
@credal_relative_likelihood_traverser.register(nn.Sequential)
def _(base: nn.Softmax | nn.Sequential, state: State) -> tuple[nn.Module, State]:
    """Skip softmax sequential layer at the end."""
    return base, state
