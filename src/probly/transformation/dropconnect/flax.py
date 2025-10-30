"""Flax dropconnect implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax.nnx import Linear, Module, Param, Rngs

from .common import register

if TYPE_CHECKING:
    from jax import Array


class DropConnectDense(Module):
   
    def __init__(
        self,
        base_layer: Linear,
        p: float = 0.25,
        *,
        rngs: Rngs | None = None,
    ) -> None:
       
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.p = p
        self.use_bias = base_layer.use_bias
        self.rngs = rngs

        # Copy weights and bias from base layer
        self.weight = Param(base_layer.kernel.value.copy())
        if self.use_bias:
            self.bias = Param(base_layer.bias.value.copy())
        else:
            self.bias = None

    def __call__(self, x: Array) -> Array:
       
        if self.rngs is not None and "dropout" in self.rngs:
            # Training mode: apply DropConnect
            mask = jax.random.bernoulli(
                self.rngs.dropout(),
                1.0 - self.p,
                shape=self.weight.value.shape,
            )
            weight = self.weight.value * mask
        else:
            # Inference mode: scale weights
            weight = self.weight.value * (1.0 - self.p)

        # Apply linear transformation
        output = jnp.dot(x, weight)
        if self.use_bias:
            output = output + self.bias.value
        return output


def replace_flax_dropconnect(obj: Linear, p: float, **kwargs) -> DropConnectDense:
   
    rngs = kwargs.get("rngs", None)
    return DropConnectDense(obj, p=p, rngs=rngs)


register(Linear, replace_flax_dropconnect)
