from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from typing import Union, Tuple, Literal, Optional


class DropConnectLinear(nnx.Module):
    """
    Linear mit DropConnect auf die Gewichte während des Trainings.
    Wrappt eine bestehende nnx.Linear-Schicht.
    """
    def __init__(self, base_layer: nnx.Linear, p: float = 0.25) -> None:
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1); got {p}")
        self.base_layer = base_layer
        self.p = float(p)

        # Reproduzierbarer RNG-State (einfach gehalten)
        self._key = jax.random.PRNGKey(0)

        # Metadaten aus der Basisschicht – robust auf weight/kernel
        self.in_features = getattr(base_layer, "in_features", None)
        self.out_features = getattr(base_layer, "out_features", None)

    def __call__(self, x: jnp.ndarray, *, training: bool | None = None) -> jnp.ndarray:
        """
        Forward-Pass mit DropConnect.
        Args:
            x: [batch, in_features]
            training: Wenn True → DropConnect aktiv; wenn False → Inference-Skalierung;
                      wenn None → fallback auf getattr(self, 'training', False)
        """
        # Gewicht/Bias robust auslesen (NNX vs linen)
        weight = getattr(self.base_layer, "weight", None)
        if weight is None:
            weight = getattr(self.base_layer, "kernel")  # linen-Kompatibilität
        bias = getattr(self.base_layer, "bias", None)

        if training is None:
            training = getattr(self, "training", False)

        if training:
            self._key, subkey = jax.random.split(self._key)
            keep_prob = 1.0 - self.p
            mask = jax.random.bernoulli(subkey, p=keep_prob, shape=weight.shape)
            # dtype an Gewicht anpassen (wichtig bei float16/bfloat16)
            mask = mask.astype(weight.dtype)
            eff_weight = weight * mask
        else:
            # Erwartungswert-Korrektur bei Inference
            eff_weight = weight * (1.0 - self.p)

        # Linearer Vorwärtslauf
        # (bewusst eigene MatMul statt base_layer(x), da wir eff_weight verwenden)
        y = jnp.matmul(x, eff_weight)
        if bias is not None:
            y = y + bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={getattr(self.base_layer, 'bias', None) is not None}, "
            f"p={self.p}"
        )
        
        
_Size2 = Union[int, Tuple[int, int]]


def _pair(v: _Size2) -> Tuple[int, int]:
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(f"Expected tuple of length 2, got {v}")
        return int(v[0]), int(v[1])
    return int(v), int(v)


class Conv2d(nnx.Module):
    """
    PyTorch-like Conv2d implemented on top of flax.nnx.Conv.

    Expects input in NCHW (N, C_in, H, W), returns NCHW (N, C_out, H_out, W_out).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size2,
        stride: _Size2 = 1,
        padding: Union[str, _Size2] = 0,
        dilation: _Size2 = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        *,
        rngs: nnx.Rngs,
        dtype=None,
        param_dtype=jnp.float32,
    ) -> None:
        if padding_mode != "zeros":
            # you *can* extend this later, but nnx.Conv already supports CIRCULAR, etc.
            # To stay close to your original pytorch semantics we keep it simple.
            raise NotImplementedError(
                f"padding_mode='{padding_mode}' is not implemented; use 'zeros'."
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

        # Handle padding argument similar to PyTorch
        self._explicit_padding: Optional[Tuple[Tuple[int, int], ...]]
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
                    f"Unsupported padding string '{padding}'. "
                    "Use 'same', 'valid', or integer/tuple."
                )
        else:
            # numeric padding → we do explicit jnp.pad + VALID conv
            p_h, p_w = _pair(padding)
            # NCHW -> pad on H and W
            # dims: (N, C, H, W)
            self._explicit_padding = (
                (0, 0),        # N
                (0, 0),        # C
                (p_h, p_h),    # H
                (p_w, p_w),    # W
            )
            flax_padding = "VALID"

        # Underlying nnx.Conv works on NHWC, so we’ll transpose in __call__
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
        """
        x: (N, C_in, H, W)  -> returns (N, C_out, H_out, W_out)
        """
        if x.ndim != 4:
            raise ValueError(f"Conv2d expects 4D input (N, C, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input has {x.shape[1]} channels, but Conv2d.in_channels={self.in_channels}"
            )

        # Apply explicit padding in NCHW if configured
        if self._explicit_padding is not None:
            x = jnp.pad(x, self._explicit_padding, mode="constant")

        # nnx.Conv expects NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # (N, H, W, C)

        y_nhwc = self.conv(x_nhwc)

        # back to NCHW
        y = jnp.transpose(y_nhwc, (0, 3, 1, 2))
        return y


# Make it visible as nnx.Conv2d for your tests & utilities
setattr(nnx, "Conv2d", Conv2d)
setattr(nnx, "DropConnectLinear", DropConnectLinear) 