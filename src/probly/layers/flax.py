"""NNX Dense + DropConnectDense layer implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flax import nnx
import jax
from jax import Array
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable


def _apply_dropconnect(
    w: Array,
    key: Array,
    p: float,
    rescale: bool = True,
) -> Array:
    """Apply DropConnect mask to weights."""
    if p <= 0.0:
        return w
    if p >= 1.0:
        return jnp.zeros_like(w)

    keep = 1.0 - p
    mask = jax.random.bernoulli(key, keep, shape=w.shape)
    out = w * mask
    return out / keep if rescale else out


class Dense(nnx.Module):
    """Minimal Dense layer implemented using flax.nnx."""

    def __init__(
        self,
        features: int,
        use_bias: bool = True,
        dtype: Any = jnp.float32,  # noqa: ANN401
        param_dtype: Any = jnp.float32,  # noqa: ANN401
        precision: Any | None = None,  # noqa: ANN401
        kernel_init: Callable[..., Array] = nnx.initializers.lecun_normal(),  # noqa: B008
        bias_init: Callable[..., Array] = nnx.initializers.zeros,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize an NNX Dense layer.

        Parameters
        ----------
        features:
            Output dimension.
        use_bias:
            Whether a learnable bias term is added.
        dtype:
            Computation dtype.
        param_dtype:
            Parameter dtype.
        precision:
            Optional JAX precision for dot products.
        kernel_init:
            Initializer for the kernel parameter.
        bias_init:
            Initializer for the bias parameter.
        rngs:
            NNX RNG container used to initialize parameters.
        """
        self.features = features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.rngs = rngs

        self.kernel = nnx.Param(
            kernel_init(rngs(), (1, features), param_dtype),
        )

        if use_bias:
            self.bias = nnx.Param(
                bias_init(rngs(), (features,), param_dtype),
            )
        else:
            self.bias = None

    def __call__(self, x: Array) -> Array:
        """Apply the linear transformation.

        Parameters
        ----------
        x : Array
            Input of shape [..., in_features].

        Returns:
        -------
        Array
            Output of shape [..., features].
        """
        x = jnp.asarray(x, self.dtype)
        y = jnp.dot(x, self.kernel.value)

        if self.use_bias and self.bias is not None:
            y = y + self.bias.value

        return y


class DropConnectDense(nnx.Module):
    """Dense layer with DropConnect regularization (NNX version)."""

    def __init__(
        self,
        features: int,
        use_bias: bool = True,
        dtype: Any = jnp.float32,  # noqa: ANN401
        param_dtype: Any = jnp.float32,  # noqa: ANN401
        precision: Any | None = None,  # noqa: ANN401
        kernel_init: Callable[..., Array] = nnx.initializers.lecun_normal(),  # noqa: B008
        bias_init: Callable[..., Array] = nnx.initializers.zeros,
        p: float = 0.25,
        rescale: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize a DropConnect dense layer for flax.nnx.

        Parameters
        ----------
        features:
            Output dimension.
        use_bias:
            Whether to include a learnable bias.
        dtype:
            Computation dtype.
        param_dtype:
            Parameter dtype.
        precision:
            Optional JAX precision for dot products.
        kernel_init:
            Initializer for the kernel.
        bias_init:
            Initializer for the bias.
        p:
            DropConnect probability.
        rescale:
            Whether to rescale surviving weights by 1/(1-p).
        rngs:
            RNG container for both params + DropConnect calls.
        """
        self.features = features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.p = p
        self.rescale = rescale
        self.rngs = rngs

        self.kernel = nnx.Param(
            kernel_init(rngs(), (1, features), param_dtype),
        )
        if use_bias:
            self.bias = nnx.Param(
                bias_init(rngs(), (features,), param_dtype),
            )
        else:
            self.bias = None

    @classmethod
    def from_dense(
        cls,
        d: Dense,
        p: float = 0.25,
        rescale: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> DropConnectDense:
        """Recreate DropConnectDense from an NNX Dense instance.

        Parameters
        ----------
        d:
            Base Dense layer to copy configuration from.
        p:
            DropConnect probability.
        rescale:
            Whether to rescale surviving weights by 1/(1-p).
        rngs:
            RNG container for new parameters.

        Returns:
        -------
        DropConnectDense
        """
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
            rngs=rngs,
        )

    def __call__(self, x: Array, *, train: bool = True) -> Array:
        """Apply the dense transformation with optional DropConnect.

        Parameters
        ----------
        x : Array
            Input array.
        train : bool
            If True, DropConnect mask is applied.

        Returns:
        -------
        Array
            Output after affine transformation ± DropConnect.
        """
        x = jnp.asarray(x, self.dtype)

        w = self.kernel.value
        if train:
            key = self.rngs()
            key = jax.random.fold_in(key, 0)
            w = _apply_dropconnect(w, key, self.p, self.rescale)

        y = jnp.dot(x, w)

        if self.use_bias and self.bias is not None:
            y = y + self.bias.value

        return y
