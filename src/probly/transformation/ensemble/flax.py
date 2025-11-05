"""Flax ensemble implementation."""

from __future__ import annotations

from flax import nnx
import jax

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, TRAVERSE_KEYS, singledispatch_traverser, traverse

from .common import register

reset_traverser = singledispatch_traverser[nnx.Module](name="reset_traverser")


@reset_traverser.register
def _(obj: nnx.Linear) -> nnx.Module:
    """Re-initialize parameters of a flax module."""
    rng = nnx.Rngs(params=jax.random.key(42))
    obj.__init__(obj.in_features, obj.out_features, rngs=rng)
    return obj


@reset_traverser.register
def _(obj: nnx.Conv) -> nnx.Module:
    """Re-initialize parameters of a flax module."""
    rng = nnx.Rngs(params=jax.random.key(42))
    obj.__init__(obj.in_features, obj.in_features, obj.kernel_size, obj.padding, rngs=rng)
    return obj


def _clone(obj: nnx.Module) -> nnx.Module:
    """Deep copy of params for flax module."""
    return traverse(obj, nn_traverser, init={CLONE: True})


def _clone_reset(obj: nnx.Module) -> nnx.Module:
    """Deep copy of params for flax module with re-initialization."""
    return traverse(obj, nn_compose(reset_traverser), init={CLONE: True, TRAVERSE_KEYS: True})


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
