"""Flax nnx ensemble implementation."""

from __future__ import annotations

from flax import nnx

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, singledispatch_traverser, traverse

from .common import register

reset_traverser = singledispatch_traverser[nnx.Module](name="reset_traverser")


@reset_traverser.register
def _(obj: nnx.Module) -> nnx.Module:
    msg = "resetting parameters of flax models is not supported yet."
    raise NotImplementedError(msg)


def _clone(obj: nnx.Module) -> nnx.Module:
    return traverse(obj, nn_traverser, init={CLONE: True})


def _clone_reset(obj: nnx.Module) -> nnx.Module:
    return traverse(obj, nn_compose(reset_traverser), init={CLONE: True})


def generate_flax_ensemble(
    obj: nnx.Module,
    num_members: int,
    reset_params: bool,
) -> nnx.List:
    """Build a flax ensemble based on :cite:`lakshminarayananSimpleScalable2017`."""
    if reset_params:
        return nnx.List([_clone_reset(obj) for _ in range(num_members)])
    return nnx.List([_clone(obj) for _ in range(num_members)])


register(nnx.Module, generate_flax_ensemble)
