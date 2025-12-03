"""Flax DropConnect and Conv2d implementations."""

from __future__ import annotations

from typing import Literal, Union

from flax import nnx
import jax
import jax.numpy as jnp


class DropConnectLinear(nnx.Module):
    """Linear layer with DropConnect on weights during training.

    Wichtig: Wir WRAPPEN keinen nnx.Linear mehr,
    sondern übernehmen nur dessen Parameter (kernel, bias).
    So tauchen keine nnx.Linear-Instanzen mehr im Modulbaum auf.
    """

    def __init__(self, base_layer: nnx.Linear, p: float = 0.25) -> None:
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1); got {p}")

        # DropConnect-Parameter
        self.p = float(p)
        self._key = jax.random.PRNGKey(0)

        # Parameter direkt übernehmen (keinen nnx.Linear mehr speichern!)
        # base_layer.kernel und base_layer.bias sind nnx.Param
        self.kernel = base_layer.kernel
        self.bias = getattr(base_layer, "bias", None)

        # Metadaten (optional, nur für repr / Debug)
        self.in_features = getattr(base_layer, "in_features", None)
        self.out_features = getattr(base_layer, "out_features", None)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        training: bool | None = None,
        rng: jax.random.KeyArray | None = None,
    ) -> jnp.ndarray:
        """Forward pass with DropConnect.

        Args:
            x: [batch, in_features]
            training:
                - True  → DropConnect aktiv
                - False → Inference mit Erwartungs-Skalierung
                - None  → fallback auf getattr(self, "training", False)
        """
        weight = self.kernel
        bias = self.bias

        if training is None:
            training = getattr(self, "training", False)

        if training:
            self._key, subkey = jax.random.split(self._key)
            keep_prob = 1.0 - self.p
            mask = jax.random.bernoulli(subkey, p=keep_prob, shape=weight.value.shape)
            mask = mask.astype(weight.value.dtype)
            eff_weight = weight.value * mask
        else:
            # Erwartungswert-Korrektur im Inference-Modus
            eff_weight = weight.value * (1.0 - self.p)

        # Linear forward pass
        y = jnp.matmul(x, eff_weight)
        if bias is not None:
            y = y + bias.value
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"p={self.p}"
        )


_Size2 = Union[int, tuple[int, int]]


def _pair(v: _Size2) -> tuple[int, int]:
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(f"Expected tuple of length 2, got {v}")
        return int(v[0]), int(v[1])
    return int(v), int(v)


class Conv2d(nnx.Module):
    """PyTorch-like Conv2d implemented on top of flax.nnx.Conv.
    Expects input in NCHW (N, C_in, H, W), returns NCHW (N, C_out, H_out, W_out).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size2,
        stride: _Size2 = 1,
        padding: str | _Size2 = 0,
        dilation: _Size2 = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if padding_mode != "zeros":
            raise NotImplementedError(
                f"padding_mode='{padding_mode}' is not implemented; use 'zeros'.",
            )

        k_h, k_w = _pair(kernel_size)
        s_h, s_w = _pair(stride)
        d_h, d_w = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.stride = (s_h, s_w)
        self.dilation = (d_h, d_w)
        self.groups = groups
        self.bias_flag = bias
        self.padding_mode = padding_mode
        self.dtype = dtype
        self.param_dtype = param_dtype

        # Handle padding argument
        self._explicit_padding: tuple[tuple[int, int], ...] | None
        if isinstance(padding, str):
            pad_str = padding.lower()
            if pad_str == "same":
                flax_padding = "SAME"
                self._explicit_padding = None
            elif pad_str == "valid":
                flax_padding = "VALID"
                self._explicit_padding = None
            else:
                raise ValueError(
                    f"Unsupported padding string '{padding}'. Use 'same', 'valid', or integer/tuple.",
                )
        else:
            p_h, p_w = _pair(padding)
            self._explicit_padding = (
                (0, 0),  # N
                (0, 0),  # C
                (p_h, p_h),  # H
                (p_w, p_w),  # W
            )
            flax_padding = "VALID"

        # Underlying nnx.Conv works on NHWC
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(k_h, k_w),
            strides=(s_h, s_w),
            padding=flax_padding,
            kernel_dilation=(d_h, d_w),
            feature_group_count=groups,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (N, C_in, H, W)  -> returns (N, C_out, H_out, W_out)"""
        if x.ndim != 4:
            raise ValueError(f"Conv2d expects 4D input (N, C, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input has {x.shape[1]} channels, but Conv2d.in_channels={self.in_channels}",
            )

        # Apply explicit padding in NCHW if configured
        if self._explicit_padding is not None:
            x = jnp.pad(x, self._explicit_padding, mode="constant")

        # nnx.Conv expects NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # (N, H, W, C)
        y_nhwc = self.conv(x_nhwc)

        # Back to NCHW
        y = jnp.transpose(y_nhwc, (0, 3, 1, 2))
        return y


# Make available as nnx.Conv2d and nnx.DropConnectLinear
nnx.Conv2d = Conv2d
nnx.DropConnectLinear = DropConnectLinear
