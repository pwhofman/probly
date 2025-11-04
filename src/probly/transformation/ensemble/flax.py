"""Flax ensemble implementation."""

from __future__ import annotations

from flax import nnx
import jax

from .common import register


def _clone(obj: nnx.Module) -> nnx.Module:
    """Deep copy of params for flax module."""
    cloned_model = nnx.clone(obj)
    return cloned_model


def _clone_reset(obj: nnx.Module, rng: jax.random.PRNGKey) -> nnx.Module:
    """Deep copy of params for flax module with re-initialization."""
    cloned_model = nnx.clone(obj)
    nnx.reseed(cloned_model, policy="match_shape", rng=rng)
    return cloned_model


def generate_flax_ensemble(obj: nnx.Module, num_members: int, reset_params: bool) -> list[nnx.Module]:
    """Build a flax ensemble by initializing n_members times."""
    if reset_params:
        return [_clone_reset(obj, jax.random.PRNGKey(i)) for i in range(num_members)]
    return [_clone(obj) for _ in range(num_members)]


register(nnx.Module, generate_flax_ensemble)
