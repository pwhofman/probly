"""Torch-backed conformal sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.conformal_set._common import (
    IntervalConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    from probly.representation.sample._common import Sample


def _ensure_torch_one_hot(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return value
        if value.dtype == torch.int64 and torch.equal(value, value.to(torch.bool)):
            return value.to(torch.bool)
    msg = "Value must be a one-hot encoded tensor of booleans or integers."
    raise ValueError(msg)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchOneHotConformalSet(TorchAxisProtected[Any], OneHotConformalSet):
    """One-hot conformal set represented as a torch tensor."""

    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}

    def __post_init__(self) -> None:
        """Validate that the tensor is a one-hot encoded tensor."""
        object.__setattr__(self, "tensor", _ensure_torch_one_hot(self.tensor))

    @classmethod
    def from_tensor_sample(cls, sample: torch.Tensor) -> Self:
        """Create a one-hot conformal set from a raw torch tensor."""
        if not isinstance(sample, torch.Tensor):
            msg = "Expected torch.Tensor for one-hot conformal sets."
            raise TypeError(msg)
        return cls(tensor=sample)

    @classmethod
    def from_sample(cls, sample: Sample[torch.Tensor]) -> Self:
        """Create a one-hot conformal set from a sample."""
        if not isinstance(sample, TorchSample):
            msg = "Expected TorchSample for one-hot conformal sets."
            raise TypeError(msg)
        return cls.from_tensor_sample(cast("torch.Tensor", sample.tensor))


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchIntervalConformalSet(TorchAxisProtected[Any], IntervalConformalSet):
    """Interval conformal set backed by a PyTorch tensor storing lower and upper bounds."""

    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}

    @classmethod
    def from_tensor_samples(cls, lower: torch.Tensor, upper: torch.Tensor) -> Self:
        """Create an interval conformal set from lower and upper bound tensors.

        Args:
            lower: The lower bound tensor.
            upper: The upper bound tensor.

        Returns:
            The created interval conformal set.
        """
        if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
            msg = "Expected torch.Tensor for interval conformal sets."
            raise TypeError(msg)
        return cls(tensor=torch.stack([lower, upper], dim=-1))

    @classmethod
    def from_samples(cls, lower: TorchSample, upper: TorchSample) -> Self:
        """Create an interval conformal set from lower and upper TorchSamples.

        Args:
            lower: The lower bound TorchSample.
            upper: The upper bound TorchSample.

        Returns:
            The created interval conformal set.
        """
        if not isinstance(lower, TorchSample) or not isinstance(upper, TorchSample):
            msg = "Expected TorchSample for interval conformal sets."
            raise TypeError(msg)
        return cls.from_tensor_samples(lower.tensor, upper.tensor)


create_onehot_conformal_set.register(torch.Tensor)(TorchOneHotConformalSet.from_tensor_sample)
create_onehot_conformal_set.register(TorchSample)(TorchOneHotConformalSet.from_sample)
create_interval_conformal_set.register(torch.Tensor)(TorchIntervalConformalSet.from_tensor_samples)
create_interval_conformal_set.register(TorchSample)(TorchIntervalConformalSet.from_samples)
