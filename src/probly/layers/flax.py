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

    def __call__(
        self,
        inputs: jax.Array,
        *,
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

        dot_general_kwargs = {}
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


class BatchEnsembleLinear(nnx.Linear):
    """Implements a BatchEnsemble Linear layer.

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
        dot_general: flax.typing.DotGeneralT, dot product function.
        promote_dtype: flax.typing.PromoteDtypeFn, function to promote the dtype of the arrays to the desired
            dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
            and a ``dtype`` keyword argument, and return a tuple of arrays with the
            promoted dtype.
        preferred_element_type: flax.typing.Dtype, Optional parameter controls the data type output by
            the dot product. This argument is passed to ``dot_general`` function.
            See ``jax.lax.dot`` for details.
        num_members: int, number of batch ensemble members.
        s: nnx.Param, rank-one factor for input features
        r: nnx.Param, rank-one factor for output features
    """

    def __init__(
        self,
        base_layer: nnx.Linear,
        rngs: nnx.Rngs | int = 1,
        num_members: int = 1,
        use_base_weights: bool = False,
        s_mean: float = 1.0,
        s_std: float = 0.01,
        r_mean: float = 1.0,
        r_std: float = 0.01,
    ) -> None:
        """Initialize a BatchEnsembleLinear layer based on a given Linear layer."""
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        if use_base_weights:
            self.kernel = base_layer.kernel
        else:
            kernel_key = rngs.params()
            kernel_init = jax.nn.initializers.lecun_normal()
            self.kernel = nnx.Param(
                kernel_init(kernel_key, (base_layer.in_features, base_layer.out_features), base_layer.param_dtype),
            )

        if base_layer.bias is not None:
            self.bias = base_layer.bias
        else:
            self.bias = nnx.data(None)

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.use_bias = base_layer.use_bias
        self.dtype = base_layer.dtype
        self.param_dtype = base_layer.param_dtype
        self.precision = base_layer.precision
        self.dot_general = base_layer.dot_general
        self.promote_dtype = base_layer.promote_dtype
        self.preferred_element_type = base_layer.preferred_element_type

        self.num_members = num_members

        s_key = rngs.params()
        s_init = s_mean + s_std * jax.random.normal(s_key, (self.num_members, base_layer.in_features))
        self.s = nnx.Param(s_init)

        r_key = rngs.params()
        r_init = r_mean + r_std * jax.random.normal(r_key, (self.num_members, base_layer.out_features))
        self.r = nnx.Param(r_init)

    def __call__(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        """Forward pass of the BatchEnsembleLinear layer.

        Args:
            inputs: jax.Array, the input of shape [B, in_features] or [E, B, in_features].
                where B is the batch size and E is the ensemble_size.

        Returns:
            jax.Array, Output of shape [E, B, out_features].
        """
        kernel = self.kernel[...]
        bias = self.bias[...] if self.bias is not None else None

        inputs, kernel, bias = self.promote_dtype((inputs, kernel, bias), dtype=self.dtype)

        dot_general_kwargs = {}
        if self.preferred_element_type is not None:
            dot_general_kwargs["preferred_element_type"] = self.preferred_element_type

        if inputs.ndim == 2:
            # If this is the first layer, expand to ensemble dimension
            inputs = jnp.expand_dims(inputs, axis=0)
            inputs = jnp.repeat(inputs, self.num_members, axis=0)
        elif inputs.ndim == 3 and inputs.shape[0] != self.num_members:
            msg = f"Expected first dim={self.num_members}, got {inputs.shape[0]}"
            raise ValueError(msg)
        # Apply s
        inputs *= self.s[:, None, :]
        # Linear transformation
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
            **dot_general_kwargs,
        )
        # Apply r
        y = y * self.r[:, None, :]
        # Add bias
        if self.use_bias:
            y += jnp.reshape(self.bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class BatchEnsembleConv(nnx.Conv):
    """Implements a BatchEnsemble convolutional layer.

    Attributes:
        kernel_shape: Sequence[int], (in_features, out_features, kernel_size)
        kernel: nnx.Param, weight matrix of the layer.
        bias: nnx.Param, bias of the layer.
        in_features: int, number of input features.
        out_features: int, number of output features.
        kernel_size: int or Sequence[int], size of the kernel.
        strides: tp.Union[None, int, tp.Sequence[int]], representing the inter-window strides.
        padding: flax.typing.PaddingLike, either the string ``'SAME'``, the string ``'VALID'``, the string
          ``'CIRCULAR'`` (periodic boundary conditions), the string `'REFLECT'`
          (reflection across the padding boundary), or a sequence of ``n``
          ``(low, high)`` integer pairs that give the padding to apply before and after each
          spatial dimension. A single int is interpreted as applying the same padding
          in all dims and passing a single int in a sequence causes the same padding
          to be used on both sides. ``'CAUSAL'`` padding for a 1D convolution will
          left-pad the convolution axis, resulting in same-sized output.
        input_dilation: tp.Union[None, int, tp.Sequence[int]], giving the
          dilation factor to apply in each spatial dimension of ``inputs``
          (default: 1). Convolution with input dilation ``d`` is equivalent to
          transposed convolution with stride ``d``.
        kernel_dilation: tp.Union[None, int, tp.Sequence[int]], giving the
          dilation factor to apply in each spatial dimension of the convolution
          kernel (default: 1). Convolution with kernel dilation
          is also known as 'atrous convolution'.
        feature_group_count: int, If specified divides the input features into groups.
        use_bias: bool, whether to add bias to the output.
        mask: typing.Optional[Array], Optional .
        dtype: typing.Optional[flax.typing.Dtype], the dtype of the computation (default: infer from input and params).
        param_dtype: flax.typing.Dtype, the dtype passed to parameter initializers.
        precision: flax.typing.PrecisionLike, numerical precision of the computation see ``jax.lax.Precision``
            for details.
        conv_general_dilated: flax.typing.DotGeneralT, dot product function.
        promote_dtype: flax.typing.PromoteDtypeFn, function to promote the dtype of the arrays to the desired
            dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
            and a ``dtype`` keyword argument, and return a tuple of arrays with the
            promoted dtype.
        preferred_element_type: flax.typing.Dtype, Optional parameter controls the data type output by
            the dot product. This argument is passed to ``dot_general`` function.
            See ``jax.lax.dot`` for details.
        num_members: int, number of batch ensemble members.
        s: nnx.Param, rank-one factor for input features.
        r: nnx.Param, rank-one factor for output features.
    """

    def __init__(
        self,
        base_layer: nnx.Conv,
        rngs: nnx.Rngs | int = 1,
        num_members: int = 1,
        use_base_weights: bool = False,
        s_mean: float = 1.0,
        s_std: float = 0.01,
        r_mean: float = 1.0,
        r_std: float = 0.01,
    ) -> None:
        """Initialize a BatchEnsembleLinear layer based on a given Linear layer."""
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.kernel_shape = base_layer.kernel_shape
        if use_base_weights:
            self.kernel = base_layer.kernel
        else:
            kernel_key = rngs.params()
            kernel_init = jax.nn.initializers.lecun_normal()
            self.kernel = nnx.Param(
                kernel_init(kernel_key, self.kernel_shape, base_layer.param_dtype),
            )

        if base_layer.bias is not None:
            self.bias = base_layer.bias
        else:
            self.bias = nnx.data(None)

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.kernel_size = base_layer.kernel_size
        self.strides = base_layer.strides
        self.padding = base_layer.padding
        self.input_dilation = base_layer.input_dilation
        self.kernel_dilation = base_layer.kernel_dilation
        self.feature_group_count = base_layer.feature_group_count
        self.use_bias = base_layer.use_bias
        self.mask = base_layer.mask
        self.dtype = base_layer.dtype
        self.param_dtype = base_layer.param_dtype
        self.precision = base_layer.precision
        self.conv_general_dilated = base_layer.conv_general_dilated
        self.promote_dtype = base_layer.promote_dtype
        self.preferred_element_type = base_layer.preferred_element_type

        self.num_members = num_members
        s_key = rngs.params()
        r_key = rngs.params()
        s_init = s_mean + s_std * jax.random.normal(s_key, (num_members, self.in_features))
        self.s = nnx.Param(s_init)
        r_init = r_mean + r_std * jax.random.normal(r_key, (num_members, self.out_features))
        self.r = nnx.Param(r_init)

    def __call__(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        """Forward pass of the BatchEnsembleConv layer.

        Args:
            inputs: jax.Array, the input of shape [B, kernel_size(n-dimensional), in_features]
                or [E, B, kernel_size(n-dimensional), in_features],
                where B is the batch size and E is the ensemble_size.

        Returns:
            jax.Array, Output of shape [E, B, kernel_size(n-dimensional), out_features].
        """
        if inputs.ndim == self.kernel.ndim:
            # If this is the first layer, expand to ensemble dimension
            inputs = jnp.repeat(inputs[None, ...], self.num_members, axis=0)
        elif inputs.ndim == (self.kernel.ndim + 1) and inputs.shape[0] != self.num_members:
            msg = f"Expected first dim={self.num_members}, got {inputs.shape[0]}"
            raise ValueError(msg)

        # flax: ensemble size, batch size, (kernel size), channel size
        s_r_dim = (slice(None),) + (None,) * (self.kernel.ndim - 1) + (slice(None),)  # ensemble size, ..., channel size
        # Apply s
        inputs *= self.s[s_r_dim]
        # Reshape to n-dimensional Convolution (ensemble_size * batch_size)
        x = inputs.reshape(inputs.shape[0] * inputs.shape[1], *inputs.shape[2:])
        # Convolutional Transformation
        y = super().__call__(x)
        # Remove bias
        if self.use_bias:
            bias = self.bias.reshape((1,) * (y.ndim - self.bias.ndim) + self.bias.shape)
            y -= bias
        # Reshape back to (ensemble_size, batch_size, (kernel_size), channel_size)
        y = y.reshape(inputs.shape[0], inputs.shape[1], *y.shape[1:])
        # Apply r
        y *= self.r[s_r_dim]
        # Add bias
        if self.use_bias:
            bias = self.bias.reshape((1,) * (y.ndim - self.bias.ndim) + self.bias.shape)
            y += bias
        return y
