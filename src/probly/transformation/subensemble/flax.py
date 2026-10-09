"""Flax subensemble implementation."""

from __future__ import annotations

from typing import Any

from flax import nnx
import jax

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, singledispatch_traverser, traverse

from ._common import subensemble_generator

reset_traverser = singledispatch_traverser[nnx.Module](name="reset_traverser")


@reset_traverser.register
def _(obj: nnx.Module) -> nnx.Module:
    if hasattr(obj, "reset_parameters"):
        obj.reset_parameters()  # ty: ignore[call-non-callable]
    return obj


def _reset_copy(module: nnx.Module) -> nnx.Module:
    return traverse(module, nn_compose(reset_traverser), init={CLONE: True})


def _copy(module: nnx.Module) -> nnx.Module:
    return traverse(module, nn_traverser, init={CLONE: True})


def _copy_layer_aware(module: Any, reset: bool) -> Any:  # noqa: ANN401
    """Copy a (possibly ``nnx.Sequential``) module, keeping plain-callable layers.

    ``nnx.Sequential`` may contain stateless callables (e.g. ``nnx.relu``,
    ``nnx.flatten``) that are not modules; they are shared as-is while module
    layers are deep-copied (and optionally reset).
    """
    fn = _reset_copy if reset else _copy
    if isinstance(module, nnx.Sequential):
        return nnx.Sequential(*[fn(layer) if isinstance(layer, nnx.Module) else layer for layer in module.layers])
    return fn(module) if isinstance(module, nnx.Module) else module


class _FrozenBackbone(nnx.Module):
    """Backbone wrapper that blocks gradient flow into the shared trunk."""

    def __init__(self, module: nnx.Module) -> None:
        self.module = module
        self.module.eval()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return jax.lax.stop_gradient(self.module(*args, **kwargs))


@subensemble_generator.register(nnx.Module)
def generate_flax_subensemble(
    obj: nnx.Module,
    num_heads: int,
    *,
    head: nnx.Module | None = None,
    reset_params: bool = True,
    head_layer: int | None = 1,
) -> nnx.List:
    """Build a flax subensemble. See :func:`probly.method.subensemble.subensemble`.

    By either:
    - splitting off the last ``head_layer`` layers of ``obj`` as the head, if no
      head model is provided (requires ``obj`` to be an ``nnx.Sequential``), or
    - using ``obj`` as the shared backbone and copying the provided ``head``
      ``num_heads`` times.

    The shared backbone is a copy of (part of) ``obj`` wrapped in
    :class:`_FrozenBackbone`, which keeps it in eval mode and stops gradients so
    only the heads train. The passed-in model is left unmodified. Non-module
    layers of an ``nnx.Sequential`` (e.g. ``nnx.relu``, ``nnx.flatten``) are
    preserved in both backbone and head.
    """
    if head is None:
        if head_layer is None:
            msg = "head_layer must be provided when head is not provided."
            raise ValueError(msg)
        if not isinstance(obj, nnx.Sequential):
            msg = (
                f"head_layer slicing is only supported for nnx.Sequential models, "
                f"but got {type(obj).__name__}. For non-Sequential models, pass "
                "an explicit head module via the `head` argument."
            )
            raise ValueError(msg)
        layers = list(obj.layers)
        if head_layer > len(layers):
            msg = f"head_layer {head_layer} must be at most {len(layers)}"
            raise ValueError(msg)
        trunk: nnx.Module = nnx.Sequential(*layers[:-head_layer])
        head = nnx.Sequential(*layers[-head_layer:])
    else:
        trunk = obj

    # Copy the trunk so freezing does not mutate the caller's model; the copied
    # layers are shared across all members.
    trunk = _copy_layer_aware(trunk, reset=False)
    heads = [_copy_layer_aware(head, reset=reset_params) for _ in range(num_heads)]

    if isinstance(trunk, nnx.Sequential):
        # One lightweight Sequential wrapper per member around the shared layers.
        members = [nnx.Sequential(_FrozenBackbone(nnx.Sequential(*trunk.layers)), h) for h in heads]
    else:
        frozen = _FrozenBackbone(trunk)
        members = [nnx.Sequential(frozen, h) for h in heads]
    return nnx.List(members)
