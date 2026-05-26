"""Measures for regression."""

from probly.lazy_types import (
    TORCH_TENSOR,
    TORCH_TENSOR_LIKE,
)

from ._common import (
    conditional_variance,
    mutual_information_variance,
    variance,
    variance_of_expected_predictive_distribution,
)
from .array import (  # noqa: F401  (registers numpy implementations)
    array_gaussian_sample_conditional_variance,
    array_gaussian_sample_mutual_information,
    array_gaussian_sample_variance_of_expected_predictive_distribution,
    array_gaussian_variance,
    array_sample_conditional_variance,
    array_sample_mutual_information,
    array_sample_variance_of_expected_predictive_distribution,
)


@variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@variance_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@conditional_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@mutual_information_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    """Register delayed implementations for torch tensors."""
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "conditional_variance",
    "mutual_information_variance",
    "variance",
    "variance_of_expected_predictive_distribution",
]
