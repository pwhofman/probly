"""Distribution subpackage."""

from probly.lazy_types import TORCH_TENSOR

from ._common import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    DirichletDistribution,
    Distribution,
    DistributionSample,
    DistributionType,
    GaussianDistribution,
    SecondOrderDistribution,
    create_categorical_distribution,
    create_categorical_distribution_from_logits,
    create_dirichlet_distribution_from_alphas,
)
from .array_categorical import ArrayCategoricalDistribution, ArrayCategoricalDistributionSample
from .array_dirichlet import ArrayDirichletDistribution
from .array_gaussian import ArrayGaussianDistribution


## Torch
@create_categorical_distribution.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_categorical as torch_categorical  # noqa: PLC0415


@create_categorical_distribution_from_logits.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_categorical as torch_categorical  # noqa: PLC0415


@create_dirichlet_distribution_from_alphas.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_dirichlet as torch_dirichlet  # noqa: PLC0415


__all__ = [
    "ArrayCategoricalDistribution",
    "ArrayCategoricalDistributionSample",
    "ArrayDirichletDistribution",
    "ArrayGaussianDistribution",
    "CategoricalDistribution",
    "CategoricalDistributionSample",
    "DirichletDistribution",
    "Distribution",
    "DistributionSample",
    "DistributionType",
    "GaussianDistribution",
    "SecondOrderDistribution",
    "create_categorical_distribution",
    "create_categorical_distribution_from_logits",
    "create_dirichlet_distribution_from_alphas",
]
