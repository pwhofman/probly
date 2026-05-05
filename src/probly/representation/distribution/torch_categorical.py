"""Torch-based categorical distribution representation."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
from probly.representation.torch_functions import torch_average

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


class TorchCategoricalDistribution(
    TorchAxisProtected[Any],
    CategoricalDistribution[torch.Tensor],
    ABC,
):
    """A categorical distribution stored as a torch tensor.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        """Get the underlying tensor representing the categorical distribution."""

    @override
    def _postprocess_protected_values(self, values: dict[str, torch.Tensor], func: Callable) -> dict[str, torch.Tensor]:
        if func in (torch.mean, torch_average):
            # Ensure mean/average of categorical distributions uses normalized probabilities.
            values["tensor"] = self.probabilities

        return values

    @override
    def with_protected_values(self, values: dict[str, Any], func: Callable | None = None) -> TorchAxisProtected[Any]:
        """Return a copy with updated protected values."""
        if func in (torch.mean, torch_average) and not isinstance(self, TorchProbabilityCategoricalDistribution):
            return TorchProbabilityCategoricalDistribution(tensor=values["tensor"])

        return super().with_protected_values(values, func)

    @property
    def _is_bernoulli(self) -> bool:
        return self.tensor.shape[-1] == 1

    def _bernoulli_tensor(self) -> torch.Tensor:
        """Get the probability or logit of the positive/last class for a Bernoulli distribution."""
        return self.tensor[..., -1]

    @override
    @property
    def unnormalized_probabilities(self) -> torch.Tensor:
        logits = self.logits
        return torch.exp(logits - torch.max(logits, dim=-1, keepdim=True).values)

    @override
    @property
    def probabilities(self) -> torch.Tensor:
        unnormalized_probabilities = self.unnormalized_probabilities
        sums = torch.sum(unnormalized_probabilities, dim=-1, keepdim=True)
        return unnormalized_probabilities / sums

    @override
    @property
    def logits(self) -> torch.Tensor:
        return torch.log(self.unnormalized_probabilities)

    @override
    @property
    def log_probabilities(self) -> torch.Tensor:
        return torch.log_softmax(self.logits, dim=-1)

    @override
    @property
    def num_classes(self) -> int:
        if self._is_bernoulli:
            return 2
        return self.unnormalized_probabilities.shape[-1]

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
    ) -> TorchSample[torch.Tensor]:
        """Sample from the categorical distribution (torch backend)."""
        flat_probabilities = self.probabilities.reshape((-1, self.num_classes))
        flat_samples = torch.multinomial(flat_probabilities, num_samples=num_samples, replacement=True, generator=rng)
        samples = flat_samples.transpose(0, 1).reshape((num_samples, *self.shape))
        return TorchSample(tensor=samples, sample_dim=0)

    @override
    def numpy(self, *, force: bool = False) -> np.ndarray:
        """Convert to a numpy array."""
        return self.probabilities.numpy(force=force)


@create_categorical_distribution.register(torch.Tensor)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchProbabilityCategoricalDistribution(TorchCategoricalDistribution):
    """A categorical distribution represented by unnormalized probabilities."""

    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}
    permitted_functions: ClassVar[set[Callable]] = {torch.mean, torch_average}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.tensor, torch.Tensor):
            msg = "probabilities must be a torch tensor."
            raise TypeError(msg)

        if self.tensor.ndim < 1:
            msg = "probabilities must have at least one dimension."
            raise ValueError(msg)

        if self._is_bernoulli:
            if torch.any(self.tensor < 0) or torch.any(self.tensor > 1):
                msg = "Bernoulli probabilities must be in the range [0, 1]."
                raise ValueError(msg)
        elif torch.any(self.tensor < 0):
            msg = "Relative probabilities must be non-negative."
            raise ValueError(msg)

    @override
    @property
    def unnormalized_probabilities(self) -> torch.Tensor:
        if self._is_bernoulli:
            p = self._bernoulli_tensor()
            q = 1 - p
            return torch.stack((q, p), dim=-1)
        return self.tensor

    @override
    def __eq__(self, value: Any) -> torch.Tensor:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, TorchCategoricalDistribution):
            eq = torch.eq(self.probabilities, value.probabilities)
        else:
            eq = torch.eq(self.tensor, value)
        return torch.all(eq, dim=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


@create_categorical_distribution_from_logits.register(torch.Tensor)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchLogitCategoricalDistribution(TorchCategoricalDistribution):
    """A categorical distribution represented by logits."""

    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}
    permitted_functions: ClassVar[set[Callable]] = {torch.mean, torch_average}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.tensor, torch.Tensor):
            msg = "logits must be a torch tensor."
            raise TypeError(msg)

        if self.tensor.ndim < 1:
            msg = "logits must have at least one dimension."
            raise ValueError(msg)

    @override
    @property
    def logits(self) -> torch.Tensor:
        if self._is_bernoulli:
            return torch.cat(
                (torch.zeros_like(self.tensor), self._bernoulli_tensor()),
                dim=-1,
            )
        return self.tensor

    @override
    def __eq__(self, value: Any) -> torch.Tensor:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, TorchCategoricalDistribution):
            eq = torch.eq(self.log_probabilities, value.log_probabilities)
        else:
            eq = torch.eq(self.tensor, value)
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
    return TorchLogitCategoricalDistribution(data)
