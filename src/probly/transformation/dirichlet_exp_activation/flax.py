"""Flax Dirichlet exp-activation implementation."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp

from probly.transformation.dirichlet_exp_activation._common import register


class _Exp(nnx.Module):
    """Elementwise exp module, turning logits into Dirichlet alpha per Malinin and Gales (2018)."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(x)


def append_activation_flax(obj: nnx.Module) -> nnx.Sequential:
    """Append exp so the model outputs Dirichlet alpha based on :cite:`malininPredictiveUncertaintyEstimation2018`.

    Unlike Sensoy's softplus + 1 parameterization, exp allows alpha values below 1,
    which the Prior Networks paper uses for very flat Dirichlets on out-of-distribution inputs.
    """
    return nnx.Sequential(obj, _Exp())


register(nnx.Module, append_activation_flax)
