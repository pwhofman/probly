"""Distribution subpackage."""

from probly.lazy_types import TORCH_TENSOR

from ._common import (
    CategoricalDistribution,
    DirichletDistribution,
    Distribution,
    DistributionType,
    GaussianDistribution,
    create_categorical_distribution,
)
from .array_categorical import ArrayCategoricalDistribution
from .array_dirichlet import ArrayDirichletDistribution
from .array_gaussian import ArrayGaussianDistribution


@create_categorical_distribution.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_categorical as torch_categorical  # noqa: PLC0415


__all__ = [
    "ArrayCategoricalDistribution",
    "ArrayDirichletDistribution",
    "ArrayGaussianDistribution",
    "CategoricalDistribution",
    "DirichletDistribution",
    "Distribution",
    "DistributionType",
    "GaussianDistribution",
    "create_categorical_distribution",
]
