"""Torch-based categorical distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution._common import CategoricalDistribution, create_categorical_distribution
from probly.representation.sample.torch import TorchTensorSample

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from probly.representation.torch_like import TorchTensorLike


@create_categorical_distribution.register(torch.Tensor)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchTensorCategoricalDistribution(
    TorchAxisProtected[Any],
    CategoricalDistribution,
):
    """A categorical distribution stored as a torch tensor.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    probabilities: torch.Tensor
    protected_axes: ClassVar[int] = 1

    @property
    def _is_bernoulli(self) -> bool:
        return self.probabilities.shape[-1] == 1

    def _bernoulli_probability(self) -> torch.Tensor:
        return self.probabilities[..., 0]

    def _normalized_probabilities(self) -> torch.Tensor:
        if self._is_bernoulli:
            msg = "Bernoulli distributions do not use categorical normalization."
            raise ValueError(msg)

        sums = torch.sum(self.probabilities, dim=-1, keepdim=True)
        if torch.any(sums <= 0):
            msg = "Relative probabilities must have strictly positive sum along the last axis."
            raise ValueError(msg)

        return self.probabilities / sums

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.probabilities, torch.Tensor):
            msg = "probabilities must be a torch tensor."
            raise TypeError(msg)

        if self.probabilities.ndim < 1:
            msg = "probabilities must have at least one dimension."
            raise ValueError(msg)

        if self._is_bernoulli:
            if torch.any(self.probabilities < 0) or torch.any(self.probabilities > 1):
                msg = "Bernoulli probabilities must be in the range [0, 1]."
                raise ValueError(msg)
        elif torch.any(self.probabilities < 0):
            msg = "Relative probabilities must be non-negative."
            raise ValueError(msg)

    @override
    def with_protected_tensor(self, tensor: torch.Tensor) -> Self:
        return type(self)(tensor)

    @override
    @property
    def num_classes(self) -> int:
        if self._is_bernoulli:
            return 2
        return self.probabilities.shape[-1]

    @override
    @property
    def entropy(self) -> torch.Tensor:
        if self._is_bernoulli:
            p = self._bernoulli_probability()
            q = 1 - p
            log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
            log_q = torch.where(q > 0, torch.log(q), torch.zeros_like(q))
            return -(p * log_p + q * log_q)

        p = self._normalized_probabilities()
        log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
        return -torch.sum(p * log_p, dim=-1)

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
    ) -> TorchTensorSample:
        """Sample from the categorical distribution (torch backend)."""
        if self._is_bernoulli:
            probabilities = self._bernoulli_probability()
            expanded = probabilities.expand((num_samples, *probabilities.shape))
            samples = torch.bernoulli(expanded, generator=rng).to(dtype=torch.int64)
            return TorchTensorSample(tensor=cast("TorchTensorLike[Any]", samples), sample_dim=0)

        flat_probabilities = self._normalized_probabilities().reshape((-1, self.num_classes))
        flat_samples = torch.multinomial(flat_probabilities, num_samples=num_samples, replacement=True, generator=rng)
        samples = flat_samples.transpose(0, 1).reshape((num_samples, *self.shape))
        return TorchTensorSample(tensor=cast("TorchTensorLike[Any]", samples), sample_dim=0)

    @override
    def numpy(self, *, force: bool = False) -> np.ndarray:
        """Convert to a numpy array."""
        return self.probabilities.numpy(force=force)

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.probabilities.__iter__()

    def __eq__(self, value: Any) -> Self:  # ty: ignore[invalid-method-override]  # noqa: ANN401, PYI032
        """Vectorized equality comparison."""
        if isinstance(value, TorchTensorCategoricalDistribution):
            return torch.eq(self.probabilities, value.probabilities)  # ty: ignore[invalid-return-type]
        return torch.eq(self.probabilities, value)  # ty: ignore[invalid-return-type]

    def __hash__(self) -> int:
        """Compute the hash of the distribution."""
        return super().__hash__()


@create_categorical_distribution.register(TorchTensorCategoricalDistribution)
def _create_torch_categorical_distribution_from_instance(
    data: TorchTensorCategoricalDistribution,
) -> TorchTensorCategoricalDistribution:
    return data
