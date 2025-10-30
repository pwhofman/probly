"""Flax ensemble implementation."""

from __future__ import annotations

from flax import nnx

from .common import register


def _reset_clone(obj: nnx.Module) -> tuple[nnx.Module, dict]:
    """New params initialized.

    Args:
        obj: nnx.Module, The model to be cloned.

    Returns:
        Tuple[nnx.Module, dict], The cloned model and its parameters.
    """
    params = nnx.clone(obj)
    return obj, params


def generate_flax_ensemble(obj: nnx.Module, n_members: int) -> list[tuple[nnx.Module, dict]]:
    """Build a flax ensemble by initializing n_members times.

    Args:
        obj: nnx.Module, The base model to be used for the ensemble.
        n_members: int, The number of members in the ensemble.

    Returns:
        List[Tuple[nnx.Module, dict]], The list of ensemble members with their parameters.
    """
    return [_reset_clone(obj) for _ in range(n_members)]


register(nnx.Module, generate_flax_ensemble)
