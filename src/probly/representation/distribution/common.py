"""Common abstractions for probability distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Protocol, Unpack

from lazy_dispatch.singledispatch import lazydispatch
from probly.representation.sampling.common_sample import Sample

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    from probly.representation.sampling.common_sample import SampleParams

type DistributionType = Literal["gaussian", "dirichlet"]


class Distribution(ABC):
    """Base class for distributions."""

    type: DistributionType

    @property
    @abstractmethod
    def entropy(self) -> float:
        """Compute entropy."""

    @abstractmethod
    def sample(self, num_samples: int) -> Sample[Any]:
        """Draw samples from Distribution."""


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


class DistributionFactory[S: Sample, D: Distribution](Protocol):
    """Factory class for creating distributions."""

    def __call__(self, sample: S) -> D:
        """Create a distribution from a sample."""


def first_dispatchable_sample(samples: Iterable, **_kwargs: Unpack[SampleParams]) -> Any:  # noqa: ANN401
    """Get the first dispatchable sample from an iterable of samples.

    Args:
        samples: The predictions to create the sample from.
        kwargs: Parameters for sample creation.

    Returns:
        The first dispatchable sample.
    """
    try:
        return samples[0]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return None


@lazydispatch[DistributionFactory, Sample](dispatch_on=first_dispatchable_sample)
def create_distribution(sample: Sample) -> Distribution:
    """Create a distribution factory from a sample."""
    msg = f"No distribution factory for sample of type {type(sample)}"
    raise NotImplementedError(msg)
