"""Conformal Prediction APS score implementation."""

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import aps_score_func


# Lazy registration - these will only be imported when needed
@aps_score_func.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@aps_score_func.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = ["aps_score_func"]
