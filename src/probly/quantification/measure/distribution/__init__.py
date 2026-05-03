"""Measures for distributions."""

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, JAX_ARRAY_SAMPLE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    LogBase,
    SecondOrderDistributionLike,
    conditional_entropy,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
    mutual_information,
)
from .array import array_categorical_entropy, array_dirichlet_entropy, array_gaussian_entropy


@entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@entropy_of_expected_predictive_distribution.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@conditional_entropy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@mutual_information.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@max_probability_complement_of_expected.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@expected_max_probability_complement.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@max_disagreement.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@entropy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_ARRAY_SAMPLE))
@entropy_of_expected_predictive_distribution.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_ARRAY_SAMPLE))
@conditional_entropy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_ARRAY_SAMPLE))
@mutual_information.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_ARRAY_SAMPLE))
@max_probability_complement_of_expected.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_ARRAY_SAMPLE))
@expected_max_probability_complement.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_ARRAY_SAMPLE))
@max_disagreement.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE, JAX_ARRAY_SAMPLE))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "LogBase",
    "SecondOrderDistributionLike",
    "array_categorical_entropy",
    "array_dirichlet_entropy",
    "array_gaussian_entropy",
    "conditional_entropy",
    "entropy",
    "entropy_of_expected_predictive_distribution",
    "expected_max_probability_complement",
    "max_disagreement",
    "max_probability_complement_of_expected",
    "mutual_information",
]
