"""Torch implementation of the Dirichlet distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, override

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution._common import DirichletDistribution, create_dirichlet_distribution_from_alphas
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.sample.torch import TorchSample
from probly.representation.torch_functions import torch_average

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


@create_dirichlet_distribution_from_alphas.register(torch.Tensor)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchDirichletDistribution(
    TorchAxisProtected[Any],
    DirichletDistribution[TorchCategoricalDistribution],
):
    """A Dirichlet distribution stored as a torch tensor.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    alphas: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"alphas": 1}
    permitted_functions: ClassVar[set[Callable]] = {torch.mean, torch.sum, torch_average}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.alphas, torch.Tensor):
            msg = "alphas must be a torch tensor."
            raise TypeError(msg)

        if self.alphas.ndim < 1:
            msg = "alphas must have at least one dimension."
            raise ValueError(msg)

        if torch.any(self.alphas <= 0):
            msg = "alphas must be strictly positive."
            raise ValueError(msg)

        if self.alphas.shape[-1] < 2:
            msg = "Dirichlet distribution requires at least 2 classes."
            raise ValueError(msg)

    @classmethod
    def from_tensor(
        cls, alphas: torch.Tensor | list[float], dtype: torch.dtype | None = None
    ) -> TorchDirichletDistribution:
        """Create a Dirichlet distribution from a tensor or list.

        Args:
            alphas: Dirichlet alpha parameters.
            dtype: Desired tensor dtype.

        Returns:
            The created torch Dirichlet distribution.
        """
        return cls(alphas=torch.as_tensor(alphas, dtype=dtype))

    @property
    def mean(self) -> TorchCategoricalDistribution:
        """Return the expected categorical probabilities."""
        return TorchProbabilityCategoricalDistribution(self.alphas / torch.sum(self.alphas, dim=-1, keepdim=True))

    @override
    def sample(self, num_samples: int = 1) -> TorchSample[TorchCategoricalDistribution]:
        """Sample categorical distributions from the Dirichlet distribution."""
        samples = torch.distributions.Dirichlet(self.alphas).sample((num_samples,))
        return TorchSample(tensor=TorchProbabilityCategoricalDistribution(samples), sample_dim=0)

    @override
    def numpy(self, *, force: bool = False) -> np.ndarray:
        """Convert the Dirichlet alpha parameters to a numpy array."""
        return self.alphas.numpy(force=force)

    @override
    def __eq__(self, value: Any) -> torch.Tensor:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, TorchDirichletDistribution):
            eq = torch.eq(self.alphas, value.alphas)
        else:
            eq = torch.eq(self.alphas, value)
        return torch.all(eq, dim=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash."""
        return object.__hash__(self)
