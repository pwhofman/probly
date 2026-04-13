"""Conformal Prediction UACQR score implementation."""

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import UACQRScore, _weight_func, uacqr_score_func


@uacqr_score_func.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@_weight_func.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@uacqr_score_func.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@_weight_func.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = ["UACQRScore", "uacqr_score_func"]
