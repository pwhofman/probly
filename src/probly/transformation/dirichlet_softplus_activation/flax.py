"""Flax Dirichlet softplus-activation implementation."""

from __future__ import annotations

from flax import nnx
import jax.nn
import jax.numpy as jnp

from probly.transformation.dirichlet_softplus_activation._common import register


class _Softplus(nnx.Module):
    """Elementwise softplus module."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.softplus(x)


class _AddOne(nnx.Module):
    """Elementwise +1 module, used to turn softplus evidence into Dirichlet alpha."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + 1


def append_activation_flax(obj: nnx.Module) -> nnx.Sequential:
    """Append Softplus + 1 so the model outputs Dirichlet alpha.

    Softplus ensures non-negative evidence; the +1 shift produces Dirichlet
    concentration parameters alpha suitable for the evidential losses in
    :mod:`probly.train.evidential.flax`.
    """
    return nnx.Sequential(obj, _Softplus(), _AddOne())


register(nnx.Module, append_activation_flax)
