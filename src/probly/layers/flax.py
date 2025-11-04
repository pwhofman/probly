"""flax layer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flax import linen as nn
import jax
from jax import Array
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.random import KeyArray

PRNG_NAME = "dropconnect"


def _apply_dropconnect(
    w: Array,
    key: KeyArray,
    p: float,
    rescale: bool = True,
) -> Array:
    """Apply DropConnect regularization to weights."""
    if p <= 0.0:
        return w
    if p >= 1.0:
        return jnp.zeros_like(w)

    keep = 1.0 - p
    mask = jax.random.bernoulli(key, keep, shape=w.shape)
    out = w * mask
    return out / keep if rescale else out


class DropConnectDense(nn.Module):
    """Dense layer with DropConnect regularization."""

    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    precision: Any | None = None
    kernel_init: Callable[..., Array] = nn.linear.default_kernel_init
    bias_init: Callable[..., Array] = nn.initializers.zeros
    p: float = 0.25
    rescale: bool = True

    @classmethod
    def from_dense(
        cls,
        d: nn.Dense,
        p: float = 0.25,
        rescale: bool = True,
    ) -> DropConnectDense:
        """Create a DropConnectDense from a regular nn.Dense."""
        return cls(
            features=d.features,
            use_bias=d.use_bias,
            dtype=d.dtype,
            param_dtype=d.param_dtype,
            precision=d.precision,
            kernel_init=d.kernel_init,
            bias_init=d.bias_init,
            p=p,
            rescale=rescale,
        )

    @nn.compact
    def __call__(self, x: Array, *, train: bool = True) -> Array:
        """Apply the layer to the input x."""
        x = jnp.asarray(x, self.dtype)
        k = self.param(
            "kernel",
            self.kernel_init,
            (x.shape[-1], self.features),
            self.param_dtype,
        )
        b = self.param("bias", self.bias_init, (self.features,), self.param_dtype) if self.use_bias else None
        w = k
        if train:
            key = self.make_rng(PRNG_NAME)
            subkey = jax.random.fold_in(key, 0)
            w = _apply_dropconnect(k, subkey, self.p, self.rescale)
        y = jnp.dot(x, w)
        if self.use_bias and b is not None:
            y = y + b
        return y
