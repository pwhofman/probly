"""Modules for computing metrics for conformal prediction."""

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    average_interval_size,
    average_set_size,
    empirical_coverage_classification,
    empirical_coverage_regression,
)


@empirical_coverage_classification.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@empirical_coverage_regression.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@average_set_size.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@average_interval_size.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@empirical_coverage_classification.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@empirical_coverage_regression.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@average_set_size.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@average_interval_size.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415
