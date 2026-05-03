"""Flax implementation of class-bias ensembles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

from ._common import BIAS_CLS, INITIALIZED, TOBIAS_VALUE, class_bias_ensemble_traverser

if TYPE_CHECKING:
    from pytraverse.core import State


@class_bias_ensemble_traverser.register(nnx.Module)
def skip_or_raise_on_non_linear(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    """Skip layers once the last linear layer is initialized; otherwise raise an error."""
    if not state[INITIALIZED]:
        msg = (
            f"class_bias_ensemble requires the last child of the model to be an nnx.Linear, found {type(obj).__name__}."
        )
        raise ValueError(msg)
    return obj, state


@class_bias_ensemble_traverser.register(nnx.Linear)
def set_class_bias(obj: nnx.Linear, state: State) -> tuple[nnx.Module, State]:
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
def skip_sequential(obj: nnx.Sequential, state: State) -> tuple[nnx.Module, State]:
    """Skip sequential containers at the end."""
    return obj, state
