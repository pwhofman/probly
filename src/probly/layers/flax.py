"""flax layer implementation."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

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


def _l2_normalize(x: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Normalize a 1D vector by its L2 norm with an ``eps`` floor."""
    norm = jnp.linalg.norm(x)
    return x / jnp.maximum(norm, eps)


def _kernel_to_matrix(kernel: jax.Array, *, is_conv: bool) -> jax.Array:
    """Reshape a layer kernel into a ``(out_features, fan_in)`` matrix for spectral norm.

    ``nnx.Linear`` stores ``(in_features, out_features)`` so we transpose to put
    ``out_features`` first. ``nnx.Conv`` stores ``(*kernel_size, in_features // groups,
    out_features)`` so we move the last axis to the front and then flatten the rest.
    """
    if is_conv:
        return jnp.moveaxis(kernel, -1, 0).reshape(kernel.shape[-1], -1)
    return kernel.T


class SpectralNormWithMultiplier(nnx.Module):
    """Apply spectral normalization with a bounded multiplier based on :cite:`liu2020SNGP`.

    The wrapped layer's kernel is reparameterized as ``weight_orig`` (a trainable
    ``nnx.Param``) with the forward pass applying the spectrally-normalized weight
    ``weight_orig * factor``, where ``factor = min(norm_multiplier / (sigma + eps), 1.0)``
    and ``sigma`` is the leading singular value estimated by power iteration. The
    wrapped module's original kernel attribute is replaced by an ``nnx.Variable``
    (mutable but not picked up by ``nnx.state(..., nnx.Param)``), so optimizer
    state filtering correctly sees a single trainable copy.

    Attributes:
        module: The wrapped layer (``nnx.Linear`` or ``nnx.Conv``).
        name: The name of the kernel parameter on the wrapped module (default ``"kernel"``).
        n_power_iterations: Number of power-iteration steps per call (default ``1``).
        norm_multiplier: Upper bound on the spectral norm multiplier (default ``1.0``).
        eps: Numerical-stability floor for division by ``sigma`` (default ``1e-12``).
        weight_orig: ``nnx.Param`` holding the original kernel as a trainable parameter.
        weight_u: ``nnx.Variable`` storing the left power-iteration vector (frozen state).
        weight_v: ``nnx.Variable`` storing the right power-iteration vector (frozen state).
    """

    def __init__(
        self,
        module: nnx.Module,
        name: str = "kernel",
        n_power_iterations: int = 1,
        norm_multiplier: float = 1.0,
        eps: float = 1e-12,
        rngs: nnx.Rngs | rnglib.RngStream | int = 1,
    ) -> None:
        """Initialize a ``SpectralNormWithMultiplier`` wrapper.

        Args:
            module: The flax module to wrap. Expected to expose ``name`` as an
                ``nnx.Param`` (default name ``"kernel"``). Linear and Conv layers are
                supported.
            name: The name of the kernel parameter on ``module`` (default ``"kernel"``).
            n_power_iterations: Number of power-iteration steps per call.
            norm_multiplier: Upper bound on the spectral norm multiplier.
            eps: Numerical-stability floor for division by ``sigma``.
            rngs: ``nnx.Rngs``, ``RngStream`` or integer seed used to initialize the
                power-iteration vectors.
        """
        if isinstance(rngs, (int, rnglib.RngStream)):
            rngs = nnx.Rngs(rngs)

        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.norm_multiplier = norm_multiplier
        self.eps = eps

        weight = getattr(module, name)
        weight_value = weight[...] if hasattr(weight, "__getitem__") else weight
        is_conv = isinstance(module, nnx.Conv)
        weight_matrix = _kernel_to_matrix(weight_value, is_conv=is_conv)

        u_key = rngs.params()
        v_key = rngs.params()
        u_init = jax.random.normal(u_key, (weight_matrix.shape[0],), dtype=weight_value.dtype)
        v_init = jax.random.normal(v_key, (weight_matrix.shape[1],), dtype=weight_value.dtype)

        self.weight_u = nnx.Variable(_l2_normalize(u_init))
        self.weight_v = nnx.Variable(_l2_normalize(v_init))
        self.weight_orig = nnx.Param(weight_value)

        # Replace the wrapped module's original kernel ``Param`` with an
        # ``nnx.Variable``. ``nnx.state(..., nnx.Param)`` filters on the strict
        # ``Param`` type, so the wrapped kernel no longer appears in the trainable
        # parameter tree. ``self.weight_orig`` (still an ``nnx.Param``) is the
        # canonical trainable copy.
        setattr(module, name, nnx.Variable(weight_value))

    def _compute_weight(self) -> jax.Array:
        """Compute the spectrally-normalized weight tensor.

        Applies one or more power iterations to refresh ``weight_u``/``weight_v``,
        scales ``weight_orig`` by ``min(norm_multiplier / sigma, 1.0)``, and
        returns the result with the kernel's original tensor layout.
        """
        weight_orig = self.weight_orig[...]
        is_conv = isinstance(self.module, nnx.Conv)
        weight_matrix = _kernel_to_matrix(weight_orig, is_conv=is_conv)
        weight_matrix_stop = jax.lax.stop_gradient(weight_matrix)

        u = self.weight_u[...]
        v = self.weight_v[...]
        for _ in range(self.n_power_iterations):
            v = _l2_normalize(jnp.matmul(weight_matrix_stop.T, u), self.eps)
            u = _l2_normalize(jnp.matmul(weight_matrix_stop, v), self.eps)

        self.weight_u[...] = jax.lax.stop_gradient(u)
        self.weight_v[...] = jax.lax.stop_gradient(v)

        sigma = jnp.dot(u, jnp.matmul(weight_matrix, v))
        factor = jnp.minimum(self.norm_multiplier / (sigma + self.eps), 1.0)
        return weight_orig * factor

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Apply the wrapped module with the spectrally-normalized weight.

        For ``nnx.Linear`` and ``nnx.Conv`` modules the forward pass is recomputed
        manually using the normalized kernel and the wrapped module's bias (if any),
        leaving the wrapped module otherwise untouched.

        Args:
            inputs: Input array consumed by the wrapped module.

        Returns:
            The output of the wrapped module computed with the normalized weight.
        """
        normalized_weight = self._compute_weight()

        if isinstance(self.module, nnx.Linear):
            return self._call_linear(inputs, normalized_weight)
        if isinstance(self.module, nnx.Conv):
            return self._call_conv(inputs, normalized_weight)

        msg = f"SpectralNormWithMultiplier supports nnx.Linear and nnx.Conv, got {type(self.module).__name__}."
        raise TypeError(msg)

    def _call_linear(self, inputs: jax.Array, kernel: jax.Array) -> jax.Array:
        """Apply a wrapped ``nnx.Linear`` with the spectrally-normalized kernel."""
        linear = cast("nnx.Linear", self.module)
        bias = linear.bias[...] if linear.bias is not None else None

        inputs_p, kernel_p, bias_p = linear.promote_dtype((inputs, kernel, bias), dtype=linear.dtype)

        dot_general_kwargs: dict[str, Any] = {}
        if linear.preferred_element_type is not None:
            dot_general_kwargs["preferred_element_type"] = linear.preferred_element_type

        out = linear.dot_general(
            inputs_p,
            kernel_p,
            (((inputs_p.ndim - 1,), (0,)), ((), ())),
            precision=linear.precision,
            **dot_general_kwargs,
        )
        if bias_p is not None:
            out = out + jnp.reshape(bias_p, (1,) * (out.ndim - 1) + (-1,))
        return out

    def _call_conv(self, inputs: jax.Array, kernel: jax.Array) -> jax.Array:
        """Apply a wrapped ``nnx.Conv`` with the spectrally-normalized kernel.

        Temporarily swaps the normalized ``kernel`` into the wrapped module's
        kernel attribute (now an ``nnx.Variable``) and restores the original
        value afterwards. ``nnx.Conv``'s call signature already handles padding,
        dilation, and feature-group bookkeeping, so reusing it avoids
        duplicating that logic.

        Note: the in-place attribute swap is not jit-friendly. Calls to this
        wrapped module are not currently jit-tested; a fully functional
        implementation would inline ``jax.lax.conv_general_dilated`` and skip
        the temporary mutation.
        """
        conv = cast("nnx.Conv", self.module)
        kernel_var = conv.kernel
        original_kernel = kernel_var[...]
        kernel_var[...] = kernel
        try:
            return conv(inputs)
        finally:
            kernel_var[...] = original_kernel


class SNGPLayer(nnx.Module):
    """Spectral-normalized Gaussian-process output layer based on :cite:`liu2020SNGP`.

    Computes random-Fourier-feature (RFF) projections of the inputs followed by
    a Bayesian linear classifier whose precision matrix is updated online via a
    momentum buffer. Returns a ``(logits, variance)`` tuple consumed by the SNGP
    Gaussian-distribution predictor.

    Attributes:
        num_inducing: Number of RFF inducing features.
        ridge_penalty: Ridge penalty added to the precision matrix at initialization.
        momentum: EMA coefficient for the precision-matrix update.
        num_classes: Number of output classes.
        W_L: ``nnx.Variable`` of shape ``(num_inducing, in_features)`` holding the
            frozen RFF projection weights.
        b_L: ``nnx.Variable`` of shape ``(num_inducing,)`` holding the frozen RFF
            phase offsets sampled uniformly in ``[0, 2 * pi)``.
        sngp: ``nnx.Linear(num_inducing, num_classes)``, the trainable Bayesian linear
            classifier.
        precision_matrix: ``nnx.Variable`` of shape ``(num_inducing, num_inducing)``,
            the running precision matrix used to compute predictive variance.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_inducing: int = 1024,
        ridge_penalty: float = 1e-6,
        momentum: float = 0.999,
        rngs: nnx.Rngs | rnglib.RngStream | int = 1,
    ) -> None:
        """Initialize an ``SNGPLayer``.

        Args:
            in_features: Number of input features.
            num_classes: Number of output classes.
            num_inducing: Number of RFF inducing features.
            ridge_penalty: Ridge penalty added to the precision matrix at initialization.
            momentum: EMA coefficient for the precision-matrix update. ``0`` accumulates.
            rngs: ``nnx.Rngs``, ``RngStream`` or integer seed used to initialize the RFF
                weights, the linear classifier, and the precision matrix.
        """
        if isinstance(rngs, (int, rnglib.RngStream)):
            rngs = nnx.Rngs(rngs)

        self.in_features = in_features
        self.num_classes = num_classes
        self.num_inducing = num_inducing
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum

        w_key = rngs.params()
        b_key = rngs.params()
        self.W_L = nnx.Variable(jax.random.normal(w_key, (num_inducing, in_features)))
        self.b_L = nnx.Variable(
            jax.random.uniform(b_key, (num_inducing,), minval=0.0, maxval=2.0 * math.pi),
        )

        self.sngp = nnx.Linear(num_inducing, num_classes, rngs=rngs)

        self.precision_matrix = nnx.Variable(jnp.eye(num_inducing) * ridge_penalty)

    def _compute_rff(self, x: jax.Array) -> jax.Array:
        """Compute random Fourier features for ``x``."""
        projection = jnp.matmul(x, self.W_L[...].T) + self.b_L[...]
        return jnp.cos(projection) * math.sqrt(2.0 / self.num_inducing)

    def _update_precision_matrix(self, phi: jax.Array, logits: jax.Array) -> None:
        """Update the running precision matrix using mean-field Laplace statistics."""
        phi_stop = jax.lax.stop_gradient(phi)
        logits_stop = jax.lax.stop_gradient(logits)
        probs = jax.nn.softmax(logits_stop, axis=-1)
        prob_variance = probs * (1.0 - probs)
        max_variance = jnp.max(prob_variance, axis=-1)

        phi_scaled = phi_stop * max_variance[..., None]
        batch_update_matrix = jnp.matmul(phi_stop.T, phi_scaled)

        precision_current = self.precision_matrix[...]
        if self.momentum > 0:
            new_precision = self.momentum * precision_current + (1.0 - self.momentum) * batch_update_matrix
        else:
            new_precision = precision_current + batch_update_matrix
        self.precision_matrix[...] = new_precision

    def __call__(
        self,
        x: jax.Array,
        *,
        update_covariance: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        """Run the SNGP forward pass.

        Args:
            x: Input of shape ``(batch_size, in_features)``.
            update_covariance: If True, refresh the precision-matrix EMA from this
                batch. Set to False at inference time. Defaults to True to mirror
                the torch backend's training-time behavior.

        Returns:
            Tuple ``(logits, variance)`` each of shape ``(batch_size, num_classes)``.
        """
        phi = self._compute_rff(x)
        logits = self.sngp(phi)

        if update_covariance:
            self._update_precision_matrix(phi, logits)

        precision_fp32 = jnp.asarray(self.precision_matrix[...], dtype=jnp.float32)
        covariance_matrix = jnp.linalg.inv(precision_fp32)
        phi_fp32 = jnp.asarray(phi, dtype=jnp.float32)
        variance = jnp.sum(phi_fp32 * jnp.matmul(phi_fp32, covariance_matrix), axis=-1)

        variance = jnp.broadcast_to(variance[..., None], logits.shape)
        return logits, variance
