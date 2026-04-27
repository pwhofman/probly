"""flax layer implementation."""

from __future__ import annotations

from typing import Any, Literal

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import first_from
import jax
import jax.numpy as jnp


def _init_fast_weight(
    key: jax.Array,
    shape: tuple[int, ...],
    init_method: Literal["random_sign", "normal"],
    mean: float,
    std: float,
    dtype: Any,  # noqa: ANN401
) -> jax.Array:
    """Sample a BatchEnsemble fast-weight tensor using random signs or Gaussian noise."""
    if init_method == "random_sign":
        signs = jax.random.bernoulli(key, p=0.5, shape=shape)
        return (signs.astype(dtype) * 2.0) - 1.0
    if init_method == "normal":
        return mean + std * jax.random.normal(key, shape, dtype=dtype)
    msg = f"Unknown init {init_method!r}; expected 'random_sign' or 'normal'."
    raise ValueError(msg)


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
        if hasattr(base_layer, "kernel_init"):
            self.kernel_init = base_layer.kernel_init
        if hasattr(base_layer, "bias_init"):
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
        if bias is not None:
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
    """BatchEnsemble Linear layer based on :cite:`wenBatchEnsemble2020`.

    The effective weight for ensemble member ``i`` is the Hadamard product ``W * (r_i s_i^T)``;
    ``r`` modulates the input features and ``s`` the output features.

    Attributes:
        kernel: nnx.Param, weight matrix of the layer.
        bias: nnx.Param of shape ``[num_members, out_features]`` (or None if the base layer had no bias).
        in_features: int, number of input features.
        out_features: int, number of output features.
        use_bias: bool, whether to add bias to the output.
        num_members: int, number of batch ensemble members.
        r: nnx.Param, rank-one factor on the input features.
        s: nnx.Param, rank-one factor on the output features.

    """

    def __init__(
        self,
        base_layer: nnx.Linear,
        rngs: nnx.Rngs | int = 1,
        num_members: int = 1,
        use_base_weights: bool = False,
        init: Literal["random_sign", "normal"] = "normal",
        r_mean: float = 1.0,
        r_std: float = 0.5,
        s_mean: float = 1.0,
        s_std: float = 0.5,
    ) -> None:
        """Initialize a BatchEnsembleLinear layer based on a given Linear layer.

        Args:
            base_layer: The base ``nnx.Linear`` layer to wrap.
            rngs: ``nnx.Rngs`` or seed used to initialize new parameters.
            num_members: Number of ensemble members.
            use_base_weights: If True, share the base layer's kernel; otherwise initialize a fresh kernel.
            init: Initialization scheme for ``r`` and ``s`` - ``"normal"`` (Gaussian, imagenet
                default) or ``"random_sign"`` ({-1, +1}, paper Appendix B).
            r_mean: Mean of the Gaussian initialization of ``r`` when ``init="normal"``.
            r_std: Standard deviation of the Gaussian initialization of ``r`` when ``init="normal"``.
            s_mean: Mean of the Gaussian initialization of ``s`` when ``init="normal"``.
            s_std: Standard deviation of the Gaussian initialization of ``s`` when ``init="normal"``.

        """
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

        if base_layer.bias is not None:
            base_bias = base_layer.bias.value
            bias_init = jnp.broadcast_to(base_bias[None, :], (num_members, self.out_features))
            self.bias = nnx.Param(bias_init)
        else:
            self.bias = nnx.data(None)

        r_key = rngs.params()
        self.r = nnx.Param(
            _init_fast_weight(r_key, (num_members, self.in_features), init, r_mean, r_std, base_layer.param_dtype),
        )
        s_key = rngs.params()
        self.s = nnx.Param(
            _init_fast_weight(s_key, (num_members, self.out_features), init, s_mean, s_std, base_layer.param_dtype),
        )

    def __call__(self, inputs: jax.Array, out_sharding: Any = None) -> jax.Array:  # noqa: ANN401
        """Forward pass of the BatchEnsembleLinear layer.

        The layer expects an input of shape ``[E * B, in_features]`` with rows
        ``[k * B, (k + 1) * B)`` belonging to ensemble member ``k``.

        Args:
            inputs: jax.Array, the input of shape ``[E * B, in_features]``.
            out_sharding: Optional sharding specification for the output array.

        Returns:
            jax.Array, Output of shape ``[E * B, out_features]``.

        """
        if inputs.ndim != 2:
            msg = f"Expected 2D input [E*B, in_features], got {inputs.ndim}D array of shape {inputs.shape}."
            raise ValueError(msg)
        eb = inputs.shape[0]
        if eb % self.num_members != 0:
            msg = f"Batch size {eb} is not divisible by num_members={self.num_members}."
            raise ValueError(msg)
        b = eb // self.num_members

        kernel = self.kernel[...]
        bias = self.bias[...] if self.bias is not None else None

        inputs, kernel, bias = self.promote_dtype((inputs, kernel, bias), dtype=self.dtype)

        dot_general_kwargs = {"out_sharding": out_sharding}
        if self.preferred_element_type is not None:
            dot_general_kwargs["preferred_element_type"] = self.preferred_element_type

        # View as [E, B, in_features] and apply r modulation
        inputs = inputs.reshape(self.num_members, b, -1)
        inputs = inputs * self.r[:, None, :]
        # Linear transformation (operates on the trailing in_features dim)
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
            **dot_general_kwargs,
        )
        # Apply s (output modulation)
        y = y * self.s[:, None, :]
        # Add per-member bias
        if bias is not None:
            y = y + bias[:, None, :]
        # Fold back to [E*B, out_features]
        return y.reshape(eb, -1)


class BatchEnsembleConv(nnx.Conv):
    """BatchEnsemble convolutional layer based on :cite:`wenBatchEnsemble2020`.

    The effective weight for ensemble member ``i`` is the Hadamard product ``W * (r_i s_i^T)``,
    realised by channel-scaling the input by ``r_i`` and the output by ``s_i`` around a
    shared convolution.

    Attributes:
        kernel_shape: Sequence[int], (kernel_size, in_features, out_features).
        kernel: nnx.Param, weight matrix of the layer.
        bias: nnx.Param of shape ``[num_members, out_features]`` (or None if the base layer had no bias).
        in_features: int, number of input features.
        out_features: int, number of output features.
        kernel_size: int or Sequence[int], size of the kernel.
        strides: int or Sequence[int], the inter-window strides.
        padding: flax.typing.PaddingLike, see ``nnx.Conv`` for the accepted forms.
        input_dilation: int or Sequence[int], dilation applied to the inputs.
        kernel_dilation: int or Sequence[int], dilation applied to the kernel ('atrous' convolution).
        feature_group_count: int, group count for grouped convolution.
        use_bias: bool, whether to add bias to the output.
        mask: optional weight mask.
        num_members: int, number of batch ensemble members.
        r: nnx.Param, rank-one factor on the input channels.
        s: nnx.Param, rank-one factor on the output channels.

    """

    def __init__(
        self,
        base_layer: nnx.Conv,
        rngs: nnx.Rngs | int = 1,
        num_members: int = 1,
        use_base_weights: bool = False,
        init: Literal["random_sign", "normal"] = "normal",
        r_mean: float = 1.0,
        r_std: float = 0.5,
        s_mean: float = 1.0,
        s_std: float = 0.5,
    ) -> None:
        """Initialize a BatchEnsembleConv layer based on a given Conv layer.

        Args:
            base_layer: The base ``nnx.Conv`` layer to wrap.
            rngs: ``nnx.Rngs`` or seed used to initialize new parameters.
            num_members: Number of ensemble members.
            use_base_weights: If True, share the base layer's kernel; otherwise initialize a fresh kernel.
            init: Initialization scheme for ``r`` and ``s`` - ``"normal"`` (Gaussian, imagenet
                default) or ``"random_sign"`` ({-1, +1}, paper Appendix B).
            r_mean: Mean of the Gaussian initialization of ``r`` when ``init="normal"``.
            r_std: Standard deviation of the Gaussian initialization of ``r`` when ``init="normal"``.
            s_mean: Mean of the Gaussian initialization of ``s`` when ``init="normal"``.
            s_std: Standard deviation of the Gaussian initialization of ``s`` when ``init="normal"``.

        """
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

        if base_layer.bias is not None:
            base_bias = base_layer.bias.value
            bias_init = jnp.broadcast_to(base_bias[None, :], (num_members, self.out_features))
            self.bias = nnx.Param(bias_init)
        else:
            self.bias = nnx.data(None)

        r_key = rngs.params()
        self.r = nnx.Param(
            _init_fast_weight(r_key, (num_members, self.in_features), init, r_mean, r_std, base_layer.param_dtype),
        )
        s_key = rngs.params()
        self.s = nnx.Param(
            _init_fast_weight(s_key, (num_members, self.out_features), init, s_mean, s_std, base_layer.param_dtype),
        )

    def __call__(
        self,
        inputs: jax.Array,
        out_sharding: Any = None,  # noqa: ANN401
    ) -> jax.Array:
        """Forward pass of the BatchEnsembleConv layer.

        The layer expects an input of shape ``[E * B, *spatial, in_features]`` with rows
        ``[k * B, (k + 1) * B)`` belonging to ensemble member ``k``.

        Args:
            inputs: jax.Array, the input of shape ``[E * B, *spatial, in_features]``.
            out_sharding: Optional sharding specification for the output array.

        Returns:
            jax.Array, Output of shape ``[E * B, *spatial_out, out_features]``.

        """
        if inputs.ndim != self.kernel.ndim:
            msg = (
                f"Expected {self.kernel.ndim}D input [E*B, *spatial, in_features], "
                f"got {inputs.ndim}D array of shape {inputs.shape}."
            )
            raise ValueError(msg)
        eb = inputs.shape[0]
        if eb % self.num_members != 0:
            msg = f"Batch size {eb} is not divisible by num_members={self.num_members}."
            raise ValueError(msg)
        b = eb // self.num_members

        # View as [E, B, *spatial, in_features] for r broadcasting
        inputs = inputs.reshape(self.num_members, b, *inputs.shape[1:])
        r_s_dim = (slice(None),) + (None,) * (self.kernel.ndim - 1) + (slice(None),)
        inputs = inputs * self.r[r_s_dim]
        # Fold back to [E*B, *spatial, in_features] for the shared convolution
        inputs = inputs.reshape(eb, *inputs.shape[2:])

        # Run the parent's convolution without its bias (we apply per-member bias post-s)
        saved_use_bias = self.use_bias
        self.use_bias = False
        try:
            y = super().__call__(inputs, out_sharding=out_sharding)
        finally:
            self.use_bias = saved_use_bias

        # View as [E, B, *spatial_out, out_features] to apply s and per-member bias
        y = y.reshape(self.num_members, b, *y.shape[1:])
        y = y * self.s[r_s_dim]
        if self.bias is not None:
            bias_dim = (slice(None),) + (None,) * (y.ndim - 2) + (slice(None),)
            y = y + self.bias[bias_dim]
        return y.reshape(eb, *y.shape[2:])
