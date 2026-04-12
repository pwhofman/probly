"""Conformal Prediction CQR score implementation."""

from ._common import cqr_score_func, CQRScore
from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, TORCH_TENSOR, TORCH_TENSOR_LIKE


@cqr_score_func.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@cqr_score_func.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415
