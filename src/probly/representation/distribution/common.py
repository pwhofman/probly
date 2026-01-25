"""Common abstractions for probability distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

type DistributionType = Literal["gaussian", "dirichlet"]


class Distribution(ABC):
    """Base class for distributions."""

    type: DistributionType

    @property
    @abstractmethod
    def entropy(self) -> float:
        """Compute entropy."""


class DirichletDistribution(Distribution):
    """Base class for Dirichlet distributions."""

    type: Literal["dirichlet"] = "dirichlet"

    @property
    @abstractmethod
    def alphas(self) -> np.ndarray:
        """Get the concentration parameters of the Dirichlet distribution."""


class GaussianDistribution(Distribution):
    """Base class for Gaussian distributions."""

    type: Literal["gaussian"] = "gaussian"

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        """Get the mean parameters."""

    @property
    @abstractmethod
    def var(self) -> np.ndarray:
        """Get the var parameters."""
