"""Distribution subpackage."""

from probly.lazy_types import TORCH_TENSOR

from ._common import (
    BernoulliDistribution,
    BernoulliDistributionSample,
    CategoricalDistribution,
    CategoricalDistributionSample,
    DirichletDistribution,
    Distribution,
    DistributionSample,
    DistributionType,
    GaussianDistribution,
    GaussianDistributionSample,
    SecondOrderDistribution,
    create_bernoulli_distribution,
    create_bernoulli_distribution_from_logits,
    create_categorical_distribution,
    create_categorical_distribution_from_logits,
    create_dirichlet_distribution_from_alphas,
    create_gaussian_distribution,
)
from .array_bernoulli import (
    ArrayBernoulliDistribution,
    ArrayBernoulliDistributionSample,
    ArrayLogitBernoulliDistribution,
    ArrayProbabilityBernoulliDistribution,
)
from .array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
    ArrayLogitCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)
from .array_dirichlet import ArrayDirichletDistribution
from .array_gaussian import ArrayGaussianDistribution, ArrayGaussianDistributionSample


## Torch
@create_categorical_distribution.delayed_register(TORCH_TENSOR)
@create_categorical_distribution_from_logits.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_categorical as torch_categorical  # noqa: PLC0415


@create_bernoulli_distribution.delayed_register(TORCH_TENSOR)
@create_bernoulli_distribution_from_logits.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_bernoulli as torch_bernoulli  # noqa: PLC0415


@create_dirichlet_distribution_from_alphas.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_dirichlet as torch_dirichlet  # noqa: PLC0415


@create_gaussian_distribution.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_gaussian as torch_gaussian  # noqa: PLC0415


__all__ = [
    "ArrayBernoulliDistribution",
    "ArrayBernoulliDistributionSample",
    "ArrayCategoricalDistribution",
    "ArrayCategoricalDistributionSample",
    "ArrayDirichletDistribution",
    "ArrayGaussianDistribution",
    "ArrayGaussianDistributionSample",
    "ArrayLogitBernoulliDistribution",
    "ArrayLogitCategoricalDistribution",
    "ArrayProbabilityBernoulliDistribution",
    "ArrayProbabilityCategoricalDistribution",
    "BernoulliDistribution",
    "BernoulliDistributionSample",
    "CategoricalDistribution",
    "CategoricalDistributionSample",
    "DirichletDistribution",
    "Distribution",
    "DistributionSample",
    "DistributionType",
    "GaussianDistribution",
    "GaussianDistributionSample",
    "SecondOrderDistribution",
    "create_bernoulli_distribution",
    "create_bernoulli_distribution_from_logits",
    "create_categorical_distribution",
    "create_categorical_distribution_from_logits",
    "create_dirichlet_distribution_from_alphas",
    "create_gaussian_distribution",
]
