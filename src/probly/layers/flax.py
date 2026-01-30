"""flax layer implementation."""

from __future__ import annotations

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import first_from
import jax
import jax.numpy as jnp


class DropConnectLinear(nnx.Module):
    """Custom Linear layer with DropConnect applied to weights during training based on :cite:`aminiDeepEvidential2020`.

    Attributes:
        kernel: nnx.Param, weight matrix of the layer.
        bias: nnx.Param, bias of the layer.
        in_features: int, number of input features.
        out_features: int, number of output features.
        use_bias: bool, whether to add bias to the output.
        dtype: typing.Optional[flax.typing.Dtype], the dtype of the computation (default: infer from input and params).
        param_dtype: flax.typing.Dtype, the dtype passed to parameter initializers.
        precision: flax.typing.PrecisionLike, numerical precision of the computation see ``jax.lax.Precision``
            for details.
        kernel_init: flax.typing.Initializer, initializer function for the weight matrix.
        bias_init: flax.typing.Initializer, initializer function for the bias.
        dot_general: flax.typing.DotGeneralT, dot product function.
        promote_dtype: flax.typing.PromoteDtypeFn, function to promote the dtype of the arrays to the desired
            dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
            and a ``dtype`` keyword argument, and return a tuple of arrays with the
            promoted dtype.
        preferred_element_type: flax.typing.Dtype, Optional parameter controls the data type output by
            the dot product. This argument is passed to ``dot_general`` function.
            See ``jax.lax.dot`` for details.
        rate: float, probability of dropping individual weights.
        deterministic: bool, if false the inputs are scaled by ``1/(1-rate)`` and
            masked, whereas if true, no mask is applied and the inputs are returned
            as is.
        rng_collection: str, the rng collection name to use when requesting a rng key.
        rngs: rnglib.Rngs or rnglib.RngStream or None, rng key.
    """

    def __init__(
        self,
        base_layer: nnx.Linear,
        rate: float = 0.25,
        *,
        rng_collection: str = "dropconnect",
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ) -> None:
        """Initialize a DropConnectLinear layer based on a given linear base layer.

        Args:
            base_layer: nnx.Linear, The original linear layer to be wrapped.
            rate: float, the dropconnect probability.
            rng_collection: str, rng collection name to use when requesting a rng key.
            rngs: nnx.Rngs or nn.RngStream or None, rng key.
        """
        self.kernel = base_layer.kernel
        self.bias = base_layer.bias if base_layer.bias is not None else None
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.use_bias = base_layer.use_bias
        self.dtype = base_layer.dtype
        self.param_dtype = base_layer.param_dtype
        self.precision = base_layer.precision
        self.kernel_init = base_layer.kernel_init
        self.bias_init = base_layer.bias_init
        self.dot_general = base_layer.dot_general
        self.promote_dtype = base_layer.promote_dtype
        self.preferred_element_type = base_layer.preferred_element_type
        self.rate = rate
        self.deterministic: bool = False
        self.rng_collection = rng_collection

        if isinstance(rngs, rnglib.Rngs):
            self.rngs = rngs[self.rng_collection].fork()
        elif isinstance(rngs, rnglib.RngStream):
            self.rngs = rngs.fork()
        elif rngs is None:
            self.rngs = nnx.data(None)
        else:
            msg = f"rngs must be a RNGS, RngStream or None, but got {type(rngs)}."
            raise TypeError(msg)

    def __call__(  # noqa: D417
        self,
        inputs: jax.Array,
        *,
        out_sharding=None,  # noqa: ANN001
        deterministic: bool = False,
        rngs: rnglib.Rngs | rnglib.RngStream | jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass of the DropConnectLinear layer.

        Args:
           inputs: jax.Array, input data.
           deterministic: bool, if false the inputs are masked, whereas if true, no mask
            is applied and the inputs are returned as is.
           rngs: nnx.Rngs, nnx.RngStream or jax.Array, optional key used to generate the dropconnect mask.

        Returns:
            jax.Array, layer output.
        """
        kernel = self.kernel[...]
        bias = self.bias[...] if self.bias is not None else None

        inputs, kernel, bias = self.promote_dtype((inputs, kernel, bias), dtype=self.dtype)

        dot_general_kwargs = {"out_sharding": out_sharding}
        if self.preferred_element_type is not None:
            dot_general_kwargs["preferred_element_type"] = self.preferred_element_type

        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to DropConnect
                as either a __call__ argument or class attribute.""",
        )

        rngs = first_from(
            rngs,
            self.rngs,
            error_msg="""No `rngs` argument was provided to DropConnect
                as either a __call__ argument or class attribute.""",
        )

        if isinstance(rngs, rnglib.Rngs):
            key = rngs[self.rng_collection]()
        elif isinstance(rngs, rnglib.RngStream):
            key = rngs()
        elif isinstance(rngs, jax.Array):
            key = rngs
        else:
            msg = f"rngs must be Rngs, RngStream or jax.Array, but got {type(rngs)}."
            raise TypeError(msg)

        if not deterministic:
            mask = jax.random.bernoulli(key=key, p=(1 - self.rate), shape=kernel.shape)
            masked_kernel = kernel * mask  # Apply DropConnect
        else:
            masked_kernel = kernel * (1 - self.rate)  # Scale weights at interference time

        out = self.dot_general(
            inputs,
            masked_kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
            **dot_general_kwargs,
        )
        if self.bias is not None:
            out += jnp.reshape(bias, (1,) * (out.ndim - 1) + (-1,))
        return out

    def set_mode(  # noqa: D417
        self,
        deterministic: bool | None = None,
        **kwargs,  # noqa: ANN003
    ) -> dict:
        """Class method used by ``nnx.set_mode``.

        Args:
            deterministic: if True, disables dropconnect masking.
        """
        if deterministic is not None:
            self.deterministic = deterministic
        return kwargs
