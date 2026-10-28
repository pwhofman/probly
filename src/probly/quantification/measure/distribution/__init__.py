"""Measures for distributions."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import (
    DEFAULT_MEAN_FIELD_FACTOR,
    DEFAULT_NUM_SAMPLES,
    LogBase,
    SecondOrderDistributionLike,
    conditional_entropy,
    dempster_shafer_uncertainty,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_generalized_entropy,
    expected_max_probability_complement,
    generalized_entropy_of_expected,
    max_disagreement,
    max_probability_complement_of_expected,
    min_expected_total_variation,
    mutual_information,
    vacuity,
)
from .numpy import (
    numpy_categorical_entropy,
    numpy_categorical_sample_min_expected_total_variation,
    numpy_dirichlet_entropy,
    numpy_dirichlet_max_probability_complement_of_expected,
    numpy_dirichlet_vacuity,
    numpy_gaussian_dempster_shafer_uncertainty,
    numpy_gaussian_entropy,
)


@entropy.delayed_register(TORCH_VALUE)
@entropy_of_expected_predictive_distribution.delayed_register(TORCH_VALUE)
@conditional_entropy.delayed_register(TORCH_VALUE)
@mutual_information.delayed_register(TORCH_VALUE)
@generalized_entropy_of_expected.delayed_register(TORCH_VALUE)
@expected_generalized_entropy.delayed_register(TORCH_VALUE)
@max_probability_complement_of_expected.delayed_register(TORCH_VALUE)
@expected_max_probability_complement.delayed_register(TORCH_VALUE)
@max_disagreement.delayed_register(TORCH_VALUE)
@min_expected_total_variation.delayed_register(TORCH_VALUE)
@vacuity.delayed_register(TORCH_VALUE)
@dempster_shafer_uncertainty.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@entropy.delayed_register(JAX_VALUE)
@entropy_of_expected_predictive_distribution.delayed_register(JAX_VALUE)
@conditional_entropy.delayed_register(JAX_VALUE)
@mutual_information.delayed_register(JAX_VALUE)
@generalized_entropy_of_expected.delayed_register(JAX_VALUE)
@expected_generalized_entropy.delayed_register(JAX_VALUE)
@max_probability_complement_of_expected.delayed_register(JAX_VALUE)
@expected_max_probability_complement.delayed_register(JAX_VALUE)
@max_disagreement.delayed_register(JAX_VALUE)
@min_expected_total_variation.delayed_register(JAX_VALUE)
@vacuity.delayed_register(JAX_VALUE)
@dempster_shafer_uncertainty.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "DEFAULT_MEAN_FIELD_FACTOR",
    "DEFAULT_NUM_SAMPLES",
    "LogBase",
    "SecondOrderDistributionLike",
    "conditional_entropy",
    "dempster_shafer_uncertainty",
    "entropy",
    "entropy_of_expected_predictive_distribution",
    "expected_generalized_entropy",
    "expected_max_probability_complement",
    "generalized_entropy_of_expected",
    "max_disagreement",
    "max_probability_complement_of_expected",
    "min_expected_total_variation",
    "mutual_information",
    "numpy_categorical_entropy",
    "numpy_categorical_sample_min_expected_total_variation",
    "numpy_dirichlet_entropy",
    "numpy_dirichlet_max_probability_complement_of_expected",
    "numpy_dirichlet_vacuity",
    "numpy_gaussian_dempster_shafer_uncertainty",
    "numpy_gaussian_entropy",
    "vacuity",
]
