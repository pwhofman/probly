"""Flax subensemble implementation."""

from __future__ import annotations

from flax import nnx

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, singledispatch_traverser, traverse

from .common import register

reset_traverser = singledispatch_traverser[nnx.Module](name="reset_traverser")


@reset_traverser.register
def _(obj: nnx.Module) -> nnx.Module:
    if hasattr(obj, "reset_parameters"):
        obj.reset_parameters()  # type: ignore[operator]
    return obj


def _reset_copy(module: nnx.Module) -> nnx.Module:
    return traverse(module, nn_compose(reset_traverser), init={CLONE: True})


def _copy(module: nnx.Module) -> nnx.Module:
    return traverse(module, nn_traverser, init={CLONE: True})


def generate_flax_subensemble(
    obj: nnx.Module,
    num_heads: int,
    reset_params: bool = False,
) -> nnx.List:
    """Build a flax subensemble by copying the last layer num_heads times, resetting the parameters of each head."""
    feature_extractor = nnx.Sequential(*obj.layers[:-1])
    last_layer = obj.layers[-1]

    if reset_params:
        heads = nnx.List([_reset_copy(last_layer) for _ in range(num_heads)])
    else:
        heads = nnx.List([_copy(last_layer) for _ in range(num_heads)])

    return nnx.List([feature_extractor, heads])


register(nnx.Module, generate_flax_subensemble)
