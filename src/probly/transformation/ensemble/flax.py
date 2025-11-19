"""Flax to Ensemble transformation."""

from __future__ import annotations

from flax.nnx import Module

from probly.traverse_nn import nn_traverser
from pytraverse import CLONE, traverse

from .common import register


def _copy_flax(model: Module) -> Module:
    """Cloned copy of the model."""
    return traverse(
        model,
        nn_traverser,
        init={CLONE: True},
    )


def generate_flax_ensemble(obj: Module, n_members: int) -> list[Module]:
    """Generate ensemble members by cloning."""
    return [_copy_flax(obj) for _ in range(n_members)]


register(Module, generate_flax_ensemble)
