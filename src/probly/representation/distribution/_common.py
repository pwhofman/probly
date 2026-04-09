"""Common abstractions for probability distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from lazy_dispatch import lazydispatch
from probly.representation.representation import Representation

if TYPE_CHECKING:
    from probly.representation.sample._common import Sample

type DistributionType = Literal["gaussian", "dirichlet", "categorical"]


class Distribution[T](Representation, ABC):
    """Base class for distributions."""

    type: DistributionType

    @property
    @abstractmethod
    def entropy(self) -> float:
        """Compute entropy."""

    @abstractmethod
    def sample(self, num_samples: int) -> Sample[T]:
        """Draw samples from Distribution."""


class CategoricalDistribution(Distribution[Any]):
    """Base class for categorical distributions."""

    type: Literal["categorical"] = "categorical"

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Get the number of classes."""


class DirichletDistribution[T: CategoricalDistribution](Distribution[T]):
    """Base class for Dirichlet distributions."""

    type: Literal["dirichlet"] = "dirichlet"

    @property
    @abstractmethod
    def alphas(self) -> Any:  # noqa: ANN401
        """Get the concentration parameters of the Dirichlet distribution."""


class GaussianDistribution[D](Distribution[D], ABC):
    """Base class for Gaussian distributions."""

    type: Literal["gaussian"] = "gaussian"

    @property
    @abstractmethod
    def mean(self) -> D:
        """Get the mean parameters."""

    @property
    @abstractmethod
    def var(self) -> D:
        """Get the var parameters."""


@lazydispatch
def create_categorical_distribution[T](data: T) -> CategoricalDistribution:
    """Create a categorical distribution from backend-specific probability data."""
    msg = f"No categorical distribution factory registered for data type {type(data)}"
    raise NotImplementedError(msg)


@create_categorical_distribution.register(CategoricalDistribution)
def _(data: CategoricalDistribution) -> CategoricalDistribution:
    """Create a categorical distribution from an instance of CategoricalDistribution."""
    return data


@lazydispatch
def create_categorical_distribution_from_logits[T](data: T) -> CategoricalDistribution:
    """Create a categorical distribution from backend-specific logit data."""
    msg = f"No categorical distribution factory from logits registered for data type {type(data)}"
    raise NotImplementedError(msg)
