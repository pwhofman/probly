"""Conformal Prediction APS score implementation."""

from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR

from ._common import APSScore as APSScore, aps_score_func


# Lazy registration - these will only be imported when needed
@aps_score_func.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@aps_score_func.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = ["aps_score_func"]
