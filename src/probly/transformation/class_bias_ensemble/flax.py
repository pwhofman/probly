"""Flax implementation of class-bias ensembles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

from ._common import BIAS_CLS, INITIALIZED, TOBIAS_VALUE, class_bias_ensemble_traverser

if TYPE_CHECKING:
    from pytraverse.core import State


@class_bias_ensemble_traverser.register(nnx.Module)
def _(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    """Skip layers if last linear layer is initialized or raise an error."""
    if not state[INITIALIZED]:
        msg = (
            f"Initialization of class-bias ensemble models "
            f"with last layer not being a linear layer is not possible. "
            f"Found last layer to be of type {type(obj)}"
        )
        raise ValueError(msg)
    return obj, state


@class_bias_ensemble_traverser.register(nnx.Linear)
def _(obj: nnx.Linear, state: State) -> tuple[nnx.Module, State]:
    """Initialize the last linear layer with class-specific bias."""
    if not state[INITIALIZED]:
        if state[BIAS_CLS] > 0:
            if obj.bias is None:
                msg = "class_bias_ensemble requires the final nnx.Linear to have a bias."
                raise ValueError(msg)
            idx = (state[BIAS_CLS] - 1) % obj.out_features
            obj.bias.value = obj.bias.value.at[idx].set(state[TOBIAS_VALUE])
        state[INITIALIZED] = True
    return obj, state


@class_bias_ensemble_traverser.register(nnx.Sequential)
def _(obj: nnx.Sequential, state: State) -> tuple[nnx.Module, State]:
    """Skip sequential containers at the end."""
    return obj, state
