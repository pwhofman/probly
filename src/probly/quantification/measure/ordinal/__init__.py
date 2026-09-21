"""Measures for ordinal classification."""

from probly.lazy_types import (
    JAX_ARRAY,
    JAX_ARRAY_LIKE,
    TORCH_TENSOR,
    TORCH_TENSOR_LIKE,
)

from ._common import (
    categorical_variance_aleatoric,
    categorical_variance_total,
    labelwise_conditional_entropy,
    labelwise_entropy,
    labelwise_entropy_of_expected_predictive_distribution,
    labelwise_expected_conditional_variance,
    labelwise_mutual_information_entropy,
    labelwise_variance,
    labelwise_variance_of_conditional_mean,
    labelwise_variance_of_expected_predictive_distribution,
    ordinal_conditional_entropy,
    ordinal_entropy,
    ordinal_entropy_of_expected_predictive_distribution,
    ordinal_expected_conditional_variance,
    ordinal_mutual_information_entropy,
    ordinal_variance,
    ordinal_variance_of_conditional_mean,
    ordinal_variance_of_expected_predictive_distribution,
)
from .numpy import (  # noqa: F401  (registers numpy implementations)
    numpy_categorical_labelwise_entropy,
    numpy_categorical_labelwise_variance,
    numpy_categorical_ordinal_entropy,
    numpy_categorical_ordinal_variance,
    numpy_categorical_sample_labelwise_conditional_entropy,
    numpy_categorical_sample_labelwise_entropy_of_expected_predictive_distribution,
    numpy_categorical_sample_labelwise_expected_conditional_variance,
    numpy_categorical_sample_labelwise_mutual_information_entropy,
    numpy_categorical_sample_labelwise_variance_of_conditional_mean,
    numpy_categorical_sample_labelwise_variance_of_expected_predictive_distribution,
    numpy_categorical_sample_ordinal_conditional_entropy,
    numpy_categorical_sample_ordinal_entropy_of_expected_predictive_distribution,
    numpy_categorical_sample_ordinal_expected_conditional_variance,
    numpy_categorical_sample_ordinal_mutual_information_entropy,
    numpy_categorical_sample_ordinal_variance_of_conditional_mean,
    numpy_categorical_sample_ordinal_variance_of_expected_predictive_distribution,
    numpy_ordinal_integer_variance_aleatoric,
    numpy_ordinal_integer_variance_total,
)


@ordinal_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_conditional_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_expected_conditional_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_entropy_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_mutual_information_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_variance_of_conditional_mean.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@ordinal_variance_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@categorical_variance_total.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@categorical_variance_aleatoric.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_entropy_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_conditional_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_mutual_information_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_variance_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_expected_conditional_variance.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@labelwise_variance_of_conditional_mean.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    """Register delayed implementations for distributions."""
    from . import torch as torch  # noqa: PLC0415


@ordinal_entropy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@ordinal_variance.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@ordinal_conditional_entropy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@ordinal_expected_conditional_variance.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@ordinal_entropy_of_expected_predictive_distribution.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@ordinal_mutual_information_entropy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@ordinal_variance_of_conditional_mean.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@ordinal_variance_of_expected_predictive_distribution.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@categorical_variance_total.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@categorical_variance_aleatoric.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@labelwise_entropy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@labelwise_variance.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@labelwise_entropy_of_expected_predictive_distribution.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@labelwise_conditional_entropy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@labelwise_mutual_information_entropy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@labelwise_variance_of_expected_predictive_distribution.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@labelwise_expected_conditional_variance.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@labelwise_variance_of_conditional_mean.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    """Register delayed implementations for distributions."""
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "categorical_variance_aleatoric",
    "categorical_variance_total",
    "labelwise_conditional_entropy",
    "labelwise_entropy",
    "labelwise_entropy_of_expected_predictive_distribution",
    "labelwise_expected_conditional_variance",
    "labelwise_mutual_information_entropy",
    "labelwise_variance",
    "labelwise_variance_of_conditional_mean",
    "labelwise_variance_of_expected_predictive_distribution",
    "ordinal_conditional_entropy",
    "ordinal_entropy",
    "ordinal_entropy_of_expected_predictive_distribution",
    "ordinal_expected_conditional_variance",
    "ordinal_mutual_information_entropy",
    "ordinal_variance",
    "ordinal_variance_of_conditional_mean",
    "ordinal_variance_of_expected_predictive_distribution",
]
