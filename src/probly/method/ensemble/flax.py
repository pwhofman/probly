"""Flax nnx ensemble implementation."""

from __future__ import annotations

from flax import nnx

from probly.traverse_nn import nn_compose, nn_traverser, reset_traverser
from pytraverse import CLONE, traverse

from ._common import ensemble_generator


def _clone(obj: nnx.Module) -> nnx.Module:
    return traverse(obj, nn_traverser, init={CLONE: True})


def _clone_reset(obj: nnx.Module) -> nnx.Module:
    return traverse(obj, nn_compose(reset_traverser), init={CLONE: True})


@ensemble_generator.register(nnx.Module)
def generate_flax_ensemble(
    obj: nnx.Module,
    num_members: int,
    reset_params: bool,
) -> nnx.List:
    """Build a flax ensemble based on :cite:`lakshminarayananSimpleScalable2017`."""
    if reset_params:
        return nnx.List([_clone_reset(obj) for _ in range(num_members)])
    return nnx.List([_clone(obj) for _ in range(num_members)])
