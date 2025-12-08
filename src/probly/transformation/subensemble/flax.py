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
    *,
    head: nnx.Module | None = None,
    reset_params: bool = False,
    head_layer: int,
) -> nnx.List:
    """Build a flax subensemble.

    By either:
    - copying head_layer (default: 1) last layers of obj num_heads, if no head model is provided.
    - using an obj as shared backbone, copying the head model num_heads times.
    Resets the parameters of each head.
    """
    # no head
    if head is None:
        head = nnx.Sequential(*obj.layers[-head_layer:])
        obj = nnx.Sequential(*obj.layers[:-head_layer])

    # obj and head
    if reset_params:
        heads = nnx.List([_reset_copy(head) for _ in range(num_heads)])
    else:
        heads = nnx.List([_copy(head) for _ in range(num_heads)])

    return nnx.List([obj, heads])


register(nnx.Module, generate_flax_subensemble)
