"""Measures for regression."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import (
    expected_conditional_variance,
    variance,
    variance_of_conditional_mean,
    variance_of_expected_predictive_distribution,
)
from .numpy import numpy_sample_expected_conditional_variance  # noqa: F401  (registers the numpy implementation)


@expected_conditional_variance.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    """Register delayed implementations for torch tensors."""
    from . import torch as torch  # noqa: PLC0415


@expected_conditional_variance.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    """Register delayed implementations for jax arrays."""
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "expected_conditional_variance",
    "variance",
    "variance_of_conditional_mean",
    "variance_of_expected_predictive_distribution",
]
