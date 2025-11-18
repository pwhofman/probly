"""Flax ensemble implementation."""

from __future__ import annotations

from flax import nnx
import jax

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, singledispatch_traverser, traverse

from .common import register

reset_traverser = singledispatch_traverser[nnx.Module](name="reset_traverser")


@reset_traverser.register
def _(obj: nnx.Module) -> nnx.Module:
    """Re-initialize parameters of a flax module."""
    rng = nnx.Rngs(params=jax.random.key(42))
    if isinstance(obj, (nnx.Conv, nnx.ConvTranspose)):
        obj.__init__(obj.in_features, obj.out_features, obj.kernel_size, rngs=rng)
    elif isinstance(obj, nnx.Linear):
        obj.__init__(obj.in_features, obj.out_features, rngs=rng)
    elif isinstance(obj, nnx.BatchNorm):
        obj.__init__(obj.axis, obj.momentum, obj.epsilon, use_running_average=False, rngs=rng)
    elif isinstance(obj, nnx.LayerNorm):
        obj.__init__(obj.num_features, obj.epsilon, rngs=rng)
    elif isinstance(obj, nnx.Embed):
        obj.__init__(obj.num_embeddings, obj.features, rngs=rng)
    elif isinstance(obj, nnx.Dropout):
        pass
    elif isinstance(obj, nnx.DropConnectLinear):
        # implementation missing
        pass
    return obj


def _clone(obj: nnx.Module) -> nnx.Module:
    """Deep copy of params for flax module."""
    return traverse(obj, nn_traverser, init={CLONE: True})


def _clone_reset(obj: nnx.Module) -> nnx.Module:
    """Deep copy of params for flax module with re-initialization."""
    return traverse(obj, nn_compose(reset_traverser), init={CLONE: True})


def generate_flax_ensemble(
    obj: nnx.Module,
    num_members: int,
    reset_params: bool,
) -> list[nnx.Module]:
    """Build a flax ensemble by initializing n_members times."""
    if reset_params:
        return [_clone_reset(obj) for _ in range(num_members)]
    return [_clone(obj) for _ in range(num_members)]


register(nnx.Module, generate_flax_ensemble)
