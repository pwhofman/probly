"""Torch-backed conformal sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Self

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.conformal_set._common import (
    ConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample.torch import TorchSample


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

    array: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    def __post_init__(self) -> None:
        """Validate that the array is a one-hot encoded tensor."""
        object.__setattr__(self, "array", _ensure_torch_one_hot(self.array))

    @classmethod
    def from_tensor_sample(cls, sample: torch.Tensor) -> Self:
        """Create a TorchOneHotConformalSet from a tensor."""
        if not isinstance(sample, torch.Tensor):
            msg = "Expected torch.Tensor for one-hot conformal sets."
            raise TypeError(msg)
        return cls(array=sample)

    @classmethod
    def from_samples(cls, sample: TorchSample) -> Self:
        """Create a TorchOneHotConformalSet from a TorchSample."""
        if not isinstance(sample, TorchSample):
            msg = "Expected TorchSample for one-hot conformal sets."
            raise TypeError(msg)
        return cls.from_tensor_sample(sample.tensor)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchIntervalConformalSet(TorchAxisProtected[Any], ConformalSet):
    """Interval conformal set backed by a PyTorch tensor storing lower and upper bounds."""

    array: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

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
        return cls(array=torch.stack([lower.flatten(), upper.flatten()], dim=-1))

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
