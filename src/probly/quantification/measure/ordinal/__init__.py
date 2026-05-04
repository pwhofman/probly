"""Measures for ordinal classification."""

from probly.lazy_types import (
    TORCH_TENSOR,
    TORCH_TENSOR_LIKE,
)

from ._common import (
    ordinal_conditional_entropy,
    ordinal_conditional_variance,
    ordinal_entropy,
    ordinal_entropy_of_expected_predictive_distribution,
    ordinal_integer_variance_aleatoric,
    ordinal_integer_variance_total,
    ordinal_mutual_information_entropy,
    ordinal_mutual_information_variance,
    ordinal_variance,
    ordinal_variance_of_expected_predictive_distribution,
)


@ordinal_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_conditional_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_conditional_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_entropy_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_mutual_information_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_mutual_information_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_variance_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_integer_variance_total.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_integer_variance_aleatoric.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    """Register delayed implementations for distributions."""
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ordinal_conditional_entropy",
    "ordinal_conditional_variance",
    "ordinal_entropy",
    "ordinal_entropy_of_expected_predictive_distribution",
    "ordinal_integer_variance_aleatoric",
    "ordinal_integer_variance_total",
    "ordinal_mutual_information_entropy",
    "ordinal_mutual_information_variance",
    "ordinal_variance",
    "ordinal_variance_of_expected_predictive_distribution",
]
