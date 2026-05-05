"""Sparse torch categorical distributions represented by grouped logits."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, override

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution._common import CategoricalDistribution, CategoricalDistributionSample
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchSparseLogCategoricalDistribution(
    TorchAxisProtected[Any],
    CategoricalDistribution[torch.Tensor],
):
    """Sparse categorical distribution with grouped log-weights.

    Shape: ``batch_shape``. The final axis of ``group_ids`` and ``logits`` stores
    sparse support entries. Entries with the same group id are combined by summing
    their exponentiated logits when converting to a dense categorical distribution.
    """

    group_ids: torch.Tensor
    logits: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"group_ids": 1, "logits": 1}

    def __post_init__(self) -> None:
        """Validate sparse log categorical fields."""
        if not isinstance(self.group_ids, torch.Tensor):
            msg = "group_ids must be a torch tensor."
            raise TypeError(msg)
        if not isinstance(self.logits, torch.Tensor):
            msg = "logits must be a torch tensor."
            raise TypeError(msg)
        if self.group_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            msg = "group_ids must be an integer tensor."
            raise TypeError(msg)
        if not torch.is_floating_point(self.logits):
            msg = "logits must be a floating point tensor."
            raise TypeError(msg)
        if self.group_ids.shape != self.logits.shape:
            msg = "group_ids and logits must have identical shapes."
            raise ValueError(msg)
        if self.group_ids.ndim < 1:
            msg = "group_ids and logits must have at least one dimension."
            raise ValueError(msg)
        if self.group_ids.shape[-1] == 0:
            msg = "sparse categorical distributions require at least one sparse entry."
            raise ValueError(msg)
        if torch.any(self.group_ids < 0):
            msg = "group_ids must be non-negative."
            raise ValueError(msg)
        if torch.any(torch.isnan(self.logits)) or torch.any(self.logits == torch.inf):
            msg = "logits must not contain NaN or positive infinity."
            raise ValueError(msg)

        self.protected_values()

    @override
    @property
    def unnormalized_probabilities(self) -> torch.Tensor:
        return self.to_dense().unnormalized_probabilities

    @override
    @property
    def probabilities(self) -> torch.Tensor:
        return self.to_dense().probabilities

    @override
    @property
    def log_probabilities(self) -> torch.Tensor:
        return self.to_dense().log_probabilities

    @override
    @property
    def num_classes(self) -> int:
        return int(torch.max(self.group_ids).item()) + 1

    def uniform_logits(self) -> TorchSparseLogCategoricalDistribution:
        """Return a copy with identical groups and uniform sparse logits.

        Returns:
            A sparse categorical distribution with shared ``group_ids`` and zero logits.
        """
        return replace(self, logits=torch.zeros_like(self.logits))

    def to_dense(self, num_classes: int | None = None) -> TorchCategoricalDistribution:
        """Convert sparse grouped logits to a dense categorical distribution.

        Args:
            num_classes: Optional dense class count. Defaults to ``max(group_ids) + 1``.

        Returns:
            Dense categorical distribution with zero mass for absent groups.
        """
        if num_classes is None:
            num_classes = self.num_classes
        if num_classes <= 0:
            msg = "num_classes must be positive."
            raise ValueError(msg)
        if torch.any(self.group_ids >= num_classes):
            msg = "num_classes must be greater than the maximum group id."
            raise ValueError(msg)

        sparse_size = self.group_ids.shape[-1]
        flat_group_ids = self.group_ids.reshape((-1, sparse_size)).to(dtype=torch.long)
        flat_logits = self.logits.reshape((-1, sparse_size))
        finite_mask = torch.isfinite(flat_logits)
        if torch.any(~torch.any(finite_mask, dim=-1)):
            msg = "Each sparse categorical distribution must contain at least one finite logit."
            raise ValueError(msg)

        shift = torch.max(flat_logits, dim=-1, keepdim=True).values
        masses = torch.exp(flat_logits - shift)
        dense = torch.zeros(
            (flat_logits.shape[0], num_classes),
            dtype=masses.dtype,
            device=masses.device,
        )
        dense.scatter_add_(dim=1, index=flat_group_ids, src=masses)
        dense = dense.reshape((*self.shape, num_classes))
        return TorchCategoricalDistribution(dense)

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
    ) -> TorchSample[torch.Tensor]:
        """Sample from the equivalent dense categorical distribution."""
        return self.to_dense().sample(num_samples=num_samples, rng=rng)

    @override
    def numpy(self, *, force: bool = False) -> np.ndarray:
        """Convert dense probabilities to a numpy array."""
        return self.to_dense().numpy(force=force)


class TorchSparseLogCategoricalDistributionSample(  # ty:ignore[conflicting-metaclass]
    CategoricalDistributionSample[TorchSparseLogCategoricalDistribution],
    TorchSample[TorchSparseLogCategoricalDistribution],
):
    """Sample type for sparse grouped log categorical distributions."""

    sample_space: ClassVar[type[CategoricalDistribution]] = TorchSparseLogCategoricalDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)
