"""Flax nnx ensemble implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax.numpy as jnp

from probly.predictor._common import predict, predict_raw
from probly.traverse_nn import nn_compose, nn_traverser, reset_traverser
from probly.traverse_nn.reset_traverser import RNGS as RESET_RNGS
from pytraverse import CLONE, traverse

from ._common import ensemble_generator

if TYPE_CHECKING:
    from flax.nnx import rnglib


def _clone(obj: nnx.Module) -> nnx.Module:
    return traverse(obj, nn_traverser, init={CLONE: True})


def _clone_reset(obj: nnx.Module, rngs: nnx.Rngs | rnglib.RngStream | int) -> nnx.Module:
    return traverse(obj, nn_compose(reset_traverser), init={CLONE: True, RESET_RNGS: rngs})


@ensemble_generator.register(nnx.Module)
def generate_flax_ensemble(
    obj: nnx.Module,
    num_members: int,
    reset_params: bool,
    rngs: nnx.Rngs | rnglib.RngStream | int = 1,
) -> nnx.List:
    """Build a flax ensemble based on :cite:`lakshminarayananSimpleScalable2017`."""
    if reset_params:
        coerced = nnx.Rngs(rngs) if isinstance(rngs, int) else rngs
        return nnx.List([_clone_reset(obj, coerced) for _ in range(num_members)])
    return nnx.List([_clone(obj) for _ in range(num_members)])


@predict_raw.register(nnx.List)
def predict_nnx_list[**In](predictor: nnx.List, *args: In.args, **kwargs: In.kwargs) -> jnp.ndarray:
    """Predict for a flax nnx list ensemble."""
    return jnp.stack([predict(p, *args, **kwargs) for p in predictor], axis=0)
