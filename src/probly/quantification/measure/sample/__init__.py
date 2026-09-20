"""Uncertainty measures for samples."""

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import mean_squared_distance_to_scaled_one_hot, sample_variance, total_logit_sample_variance
from .numpy import numpy_mean_squared_distance_to_scaled_one_hot, numpy_total_logit_sample_variance


@mean_squared_distance_to_scaled_one_hot.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@total_logit_sample_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@mean_squared_distance_to_scaled_one_hot.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@total_logit_sample_variance.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "mean_squared_distance_to_scaled_one_hot",
    "numpy_mean_squared_distance_to_scaled_one_hot",
    "numpy_total_logit_sample_variance",
    "sample_variance",
    "total_logit_sample_variance",
]
