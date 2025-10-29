"""Flax ensemble implementation."""

from __future__ import annotations

from flax.nnx import Module

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, lazydispatch_traverser, traverse

from .common import register

reset_traverser = lazydispatch_traverser[object](name="reset_traverser")


@reset_traverser.register
def _(obj: Module) -> Module:
    if hasattr(obj, "reset_parameters"):
        obj.reset_parameters()  # type: ignore[operator]
    return obj


def _reset_copy(module: Module) -> Module:
    return traverse(module, nn_compose(reset_traverser), init={CLONE: True})


def generate_flax_ensemble(obj: Module, n_members: int) -> list[Module]:
    """Build a flax ensemble by copying the base model n_members times."""
    return [_reset_copy(obj) for _ in range(n_members)]


register(Module, generate_flax_ensemble)
