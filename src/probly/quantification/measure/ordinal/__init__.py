"""Measures for ordinal classification."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

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
    numpy_ordinal_integer_variance_aleatoric,
    numpy_ordinal_integer_variance_total,
)


@categorical_variance_total.delayed_register(TORCH_VALUE)
@categorical_variance_aleatoric.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    """Register delayed implementations for distributions."""
    from . import torch as torch  # noqa: PLC0415


@categorical_variance_total.delayed_register(JAX_VALUE)
@categorical_variance_aleatoric.delayed_register(JAX_VALUE)
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
