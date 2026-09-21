"""Measures for regression."""

from probly.lazy_types import (
    JAX_ARRAY,
    JAX_ARRAY_LIKE,
    TORCH_TENSOR,
    TORCH_TENSOR_LIKE,
)

from ._common import (
    expected_conditional_variance,
    variance,
    variance_of_conditional_mean,
    variance_of_expected_predictive_distribution,
)
from .numpy import (  # noqa: F401  (registers numpy implementations)
    numpy_gaussian_sample_expected_conditional_variance,
    numpy_gaussian_sample_variance_of_conditional_mean,
    numpy_gaussian_sample_variance_of_expected_predictive_distribution,
    numpy_gaussian_variance,
    numpy_sample_expected_conditional_variance,
    numpy_sample_variance_of_conditional_mean,
    numpy_sample_variance_of_expected_predictive_distribution,
)


@variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@variance_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@expected_conditional_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@variance_of_conditional_mean.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    """Register delayed implementations for torch tensors."""
    from . import torch as torch  # noqa: PLC0415


@variance.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@variance_of_expected_predictive_distribution.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@expected_conditional_variance.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@variance_of_conditional_mean.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    """Register delayed implementations for jax arrays."""
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "expected_conditional_variance",
    "variance",
    "variance_of_conditional_mean",
    "variance_of_expected_predictive_distribution",
]
