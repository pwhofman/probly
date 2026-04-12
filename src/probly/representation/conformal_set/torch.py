"""Torch-backed conformal sets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self, override

import torch

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.conformal_set._common import (
    ConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample._common import Sample
from probly.representation.sample.torch import TorchTensorSample


def _ensure_torch_one_hot(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return value
        elif value.dtype == torch.int64:
            if torch.equal(value, value.to(torch.bool)):
                return value.to(torch.bool)
    msg = "Value must be a one-hot encoded tensor of booleans or integers."
    raise ValueError(msg)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchOneHotConformalSet(TorchAxisProtected[TorchTensorSample], OneHotConformalSet):
    array: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    def __post_init__(self) -> None:
        object.__setattr__(self, "array", _ensure_torch_one_hot(self.array))

    @classmethod
    def from_sample(cls, sample: torch.Tensor) -> Self:
        if not isinstance(sample, torch.Tensor):
            msg = "Expected torch.Tensor for one-hot conformal sets."
            raise TypeError(msg)
        return cls(array=sample)

    @classmethod
    def from_tensor_sample(cls, sample: TorchTensorSample) -> Self:
        if not isinstance(sample, TorchTensorSample):
            msg = "Expected TorchTensorSample for one-hot conformal sets."
            raise TypeError(msg)
        return cls.from_sample(sample.tensor)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchIntervalConformalSet(TorchAxisProtected[TorchTensorSample], ConformalSet):
    array: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    @classmethod
    def from_sample(cls, lower: torch.Tensor, upper: torch.Tensor) -> Self:
        if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
            msg = "Expected torch.Tensor for interval conformal sets."
            raise TypeError(msg)
        return cls(array=torch.stack([lower.flatten(), upper.flatten()], dim=-1))

    @classmethod
    def from_tensor_sample(cls, lower: TorchTensorSample, upper: TorchTensorSample) -> Self:
        if not isinstance(lower, TorchTensorSample) or not isinstance(upper, TorchTensorSample):
            msg = "Expected TorchTensorSample for interval conformal sets."
            raise TypeError(msg)
        return cls.from_sample(lower.tensor, upper.tensor)


create_onehot_conformal_set.register(torch.Tensor)(TorchOneHotConformalSet.from_sample)
create_onehot_conformal_set.register(TorchTensorSample)(TorchOneHotConformalSet.from_tensor_sample)
create_interval_conformal_set.register(torch.Tensor)(TorchIntervalConformalSet.from_sample)
create_interval_conformal_set.register(TorchTensorSample)(TorchIntervalConformalSet.from_tensor_sample)
