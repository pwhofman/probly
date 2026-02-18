"""Flax subensemble implementation."""

from __future__ import annotations

from flax import nnx

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, singledispatch_traverser, traverse

from .common import register

reset_traverser = singledispatch_traverser[nnx.Module](name="reset_traverser")
subensemble_traverser = singledispatch_traverser[nnx.Module](name="subensemble_traverser")


@subensemble_traverser.register
def _(obj: nnx.Module) -> nnx.List[nnx.Module]:
    children = nnx.List()
    for _, child in obj.iter_modules():
        if hasattr(child, "in_features"):  # type: ignore[attr-defined]
            children.append(child)
    return children


@reset_traverser.register
def _(obj: nnx.Module) -> nnx.Module:  # type: ignore[call-arg]
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
    layers = [m for m in traverse(obj, subensemble_traverser) if isinstance(m, nnx.Module)]

    if head_layer > len(layers):
        msg = f"head_layer {head_layer} must be less than to {len(layers)}"
        raise ValueError(msg)

    # no head
    if head is None:
        backbone = nnx.Sequential(*layers[:-head_layer])
        head = nnx.Sequential(*layers[-head_layer:])
    else:
        backbone = obj

    # obj and head
    if reset_params:
        heads = nnx.List([_reset_copy(head) for _ in range(num_heads)])
    else:
        heads = nnx.List([_copy(head) for _ in range(num_heads)])

    # freeze backbone
    backbone.eval()

    backbone_layers = [m for m in traverse(backbone, subensemble_traverser) if isinstance(m, nnx.Module)]

    subensemble = nnx.List(
        [
            nnx.Sequential(
                nnx.Sequential(*backbone_layers),
                nnx.Sequential(*[m for m in traverse(h, subensemble_traverser) if isinstance(m, nnx.Module)]),
            )
            for h in heads
        ],
    )
    return subensemble


register(nnx.Module, generate_flax_subensemble)
