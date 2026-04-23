"""Torch-based categorical distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, override

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution._common import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    create_categorical_distribution,
    create_categorical_distribution_from_logits,
)
from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np


@create_categorical_distribution.register(torch.Tensor)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchCategoricalDistribution(
    TorchAxisProtected[Any],
    CategoricalDistribution[torch.Tensor],
):
    """A categorical distribution stored as a torch tensor.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    unnormalized_probabilities: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"unnormalized_probabilities": 1}
    permitted_functions: ClassVar[set[Callable]] = {torch.mean}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.unnormalized_probabilities, torch.Tensor):
            msg = "probabilities must be a torch tensor."
            raise TypeError(msg)

        if self.unnormalized_probabilities.ndim < 1:
            msg = "probabilities must have at least one dimension."
            raise ValueError(msg)

        if self._is_bernoulli:
            if torch.any(self.unnormalized_probabilities < 0) or torch.any(self.unnormalized_probabilities > 1):
                msg = "Bernoulli probabilities must be in the range [0, 1]."
                raise ValueError(msg)
        elif torch.any(self.unnormalized_probabilities < 0):
            msg = "Relative probabilities must be non-negative."
            raise ValueError(msg)

    @property
    def _is_bernoulli(self) -> bool:
        return self.unnormalized_probabilities.shape[-1] == 1

    def _bernoulli_probability(self) -> torch.Tensor:
        return self.unnormalized_probabilities[..., 0]

    @override
    @property
    def probabilities(self) -> torch.Tensor:
        if self._is_bernoulli:
            p = self._bernoulli_probability()
            q = 1 - p
            return torch.stack((p, q), dim=-1)

        sums = torch.sum(self.unnormalized_probabilities, dim=-1, keepdim=True)

        return self.unnormalized_probabilities / sums

    @override
    @property
    def num_classes(self) -> int:
        if self._is_bernoulli:
            return 2
        return self.unnormalized_probabilities.shape[-1]

    @override
    @property
    def entropy(self) -> torch.Tensor:
        if self._is_bernoulli:
            p = self._bernoulli_probability()
            q = 1 - p
            log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
            log_q = torch.where(q > 0, torch.log(q), torch.zeros_like(q))
            return -(p * log_p + q * log_q)

        p = self.probabilities
        log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
        return -torch.sum(p * log_p, dim=-1)

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
    ) -> TorchSample[torch.Tensor]:
        """Sample from the categorical distribution (torch backend)."""
        if self._is_bernoulli:
            probabilities = self._bernoulli_probability()
            expanded = probabilities.expand((num_samples, *probabilities.shape))
            samples = torch.bernoulli(expanded, generator=rng).to(dtype=torch.int64)
            return TorchSample(tensor=samples, sample_dim=0)

        flat_probabilities = self.probabilities.reshape((-1, self.num_classes))
        flat_samples = torch.multinomial(flat_probabilities, num_samples=num_samples, replacement=True, generator=rng)
        samples = flat_samples.transpose(0, 1).reshape((num_samples, *self.shape))
        return TorchSample(tensor=samples, sample_dim=0)

    @override
    def numpy(self, *, force: bool = False) -> np.ndarray:
        """Convert to a numpy array."""
        return self.probabilities.numpy(force=force)

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.probabilities.__iter__()

    @override
    def __eq__(self, value: Any) -> torch.Tensor:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, TorchCategoricalDistribution):
            eq = torch.eq(self.probabilities, value.probabilities)
        else:
            eq = torch.eq(self.unnormalized_probabilities, value)
        return torch.all(eq, dim=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


class TorchCategoricalDistributionSample(  # ty:ignore[conflicting-metaclass]
    CategoricalDistributionSample[TorchCategoricalDistribution],
    TorchSample[TorchCategoricalDistribution],
):
    """Sample type for empirical second-order categorical distributions."""

    sample_space: ClassVar[type[CategoricalDistribution]] = TorchCategoricalDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


@create_categorical_distribution.register(TorchCategoricalDistribution)
def _create_torch_categorical_distribution_from_instance(
    data: TorchCategoricalDistribution,
) -> TorchCategoricalDistribution:
    return data


@create_categorical_distribution_from_logits.register(torch.Tensor)
def _create_torch_categorical_distribution_from_logits(
    data: torch.Tensor,
) -> TorchCategoricalDistribution:
    return TorchCategoricalDistribution(torch.softmax(data, dim=-1))
