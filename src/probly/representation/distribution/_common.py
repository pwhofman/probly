"""Common abstractions for probability distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from probly.representation.sample._common import Sample

type DistributionType = Literal["gaussian", "dirichlet", "categorical"]


class Distribution[T](ABC):
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


class DirichletDistribution[T](Distribution[T]):
    """Base class for Dirichlet distributions."""

    type: Literal["dirichlet"] = "dirichlet"

    @property
    @abstractmethod
    def alphas(self) -> Any:  # noqa: ANN401
        """Get the concentration parameters of the Dirichlet distribution."""


class GaussianDistribution[D](Distribution[D]):
    """Base class for Gaussian distributions."""

    type: Literal["gaussian"] = "gaussian"

    @property
    @abstractmethod
    def mean(self) -> Any:  # noqa: ANN401
        """Get the mean parameters."""

    @property
    @abstractmethod
    def var(self) -> Any:  # noqa: ANN401
        """Get the var parameters."""
