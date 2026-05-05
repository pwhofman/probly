"""Measures for ordinal classification."""

from probly.lazy_types import (
    TORCH_TENSOR,
    TORCH_TENSOR_LIKE,
)

from ._common import (
    labelwise_conditional_entropy,
    labelwise_conditional_variance,
    labelwise_entropy,
    labelwise_entropy_of_expected_predictive_distribution,
    labelwise_mutual_information_entropy,
    labelwise_mutual_information_variance,
    labelwise_variance,
    labelwise_variance_of_expected_predictive_distribution,
    ordinal_conditional_entropy,
    ordinal_conditional_variance,
    ordinal_entropy,
    ordinal_entropy_of_expected_predictive_distribution,
    categorical_variance_aleatoric,
    categorical_variance_total,
    ordinal_mutual_information_entropy,
    ordinal_mutual_information_variance,
    ordinal_variance,
    ordinal_variance_of_expected_predictive_distribution,
)
from .array import (  # noqa: F401  (registers numpy implementations)
    array_categorical_labelwise_entropy,
    array_categorical_labelwise_variance,
    array_categorical_ordinal_entropy,
    array_categorical_ordinal_variance,
    array_categorical_sample_labelwise_conditional_entropy,
    array_categorical_sample_labelwise_conditional_variance,
    array_categorical_sample_labelwise_entropy_of_expected_predictive_distribution,
    array_categorical_sample_labelwise_mutual_information_entropy,
    array_categorical_sample_labelwise_mutual_information_variance,
    array_categorical_sample_labelwise_variance_of_expected_predictive_distribution,
    array_categorical_sample_ordinal_conditional_entropy,
    array_categorical_sample_ordinal_conditional_variance,
    array_categorical_sample_ordinal_entropy_of_expected_predictive_distribution,
    array_categorical_sample_ordinal_mutual_information_entropy,
    array_categorical_sample_ordinal_mutual_information_variance,
    array_categorical_sample_ordinal_variance_of_expected_predictive_distribution,
    array_ordinal_integer_variance_aleatoric,
    array_ordinal_integer_variance_total,
)


@ordinal_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_conditional_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_conditional_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_entropy_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_mutual_information_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_mutual_information_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_variance_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@categorical_variance_total.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@categorical_variance_aleatoric.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_entropy_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_conditional_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_mutual_information_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_variance_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_conditional_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_mutual_information_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    """Register delayed implementations for distributions."""
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "labelwise_conditional_entropy",
    "labelwise_conditional_variance",
    "labelwise_entropy",
    "labelwise_entropy_of_expected_predictive_distribution",
    "labelwise_mutual_information_entropy",
    "labelwise_mutual_information_variance",
    "labelwise_variance",
    "labelwise_variance_of_expected_predictive_distribution",
    "ordinal_conditional_entropy",
    "ordinal_conditional_variance",
    "ordinal_entropy",
    "ordinal_entropy_of_expected_predictive_distribution",
    "categorical_variance_aleatoric",
    "categorical_variance_total",
    "ordinal_mutual_information_entropy",
    "ordinal_mutual_information_variance",
    "ordinal_variance",
    "ordinal_variance_of_expected_predictive_distribution",
]
