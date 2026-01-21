"""Common abstractions for probability distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    # from probly.representation.sampling.array_sample import ArraySample wenn auf stand von probly main

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

    @abstractmethod
    def sample(
        self,
        size: int,
    ) -> np.ndarray:  # müssen main aktualisieren dann sollte da als Rückgabewert ArraySample stehen
        """Draw samples from the distribution."""

    '''@classmethod
    @abstractmethod
    def from_parameters(
        cls,
        *args,
        dtype: DTypeLike | None = None,
        **kwargs,
    ) -> Distribution:
        """Create a distribution from its natural parameters."""'''
