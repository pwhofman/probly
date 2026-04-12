"""NumPy-backed conformal sets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self, override

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.conformal_set._common import (
    ConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample._common import Sample
from probly.representation.sample.array import ArraySample


def _ensure_array_one_hot(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == bool:
            return value.astype(bool)
        elif value.dtype == int:
            if np.array_equal(value, value.astype(bool)):
                return value.astype(bool)
    msg = "Value must be a one-hot encoded array of booleans or integers."
    raise ValueError(msg)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayOneHotConformalSet(ArrayAxisProtected[ArraySample], OneHotConformalSet):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    def __post_init__(self) -> None:
        object.__setattr__(self, "array", _ensure_array_one_hot(self.array))

    @classmethod
    def from_sample(cls, sample: np.ndarray) -> Self:
        if not isinstance(sample, np.ndarray):
            msg = "Expected np.ndarray for one-hot conformal sets."
            raise TypeError(msg)
        return cls(array=sample)

    @classmethod
    def from_array_sample(cls, sample: ArraySample) -> Self:
        if not isinstance(sample, ArraySample):
            msg = "Expected ArraySample for interval conformal sets."
            raise TypeError(msg)
        return cls.from_sample(sample.array)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayIntervalConformalSet(ArrayAxisProtected[ArraySample], ConformalSet):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    @classmethod
    def from_sample(cls, lower: np.ndarray, upper: np.ndarray) -> Self:
        if not isinstance(lower, np.ndarray) or not isinstance(upper, np.ndarray):
            msg = "Expected np.ndarray for interval conformal sets."
            raise TypeError(msg)
        return cls(array=np.stack([lower.flatten(), upper.flatten()], axis=-1))

    @classmethod
    def from_array_sample(cls, lower: ArraySample, upper: ArraySample) -> Self:
        if not isinstance(lower, ArraySample) or not isinstance(upper, ArraySample):
            msg = "Expected ArraySample for interval conformal sets."
            raise TypeError(msg)
        return cls.from_sample(lower.array, upper.array)


create_onehot_conformal_set.register(np.ndarray)(ArrayOneHotConformalSet.from_sample)
create_onehot_conformal_set.register(ArraySample)(ArrayOneHotConformalSet.from_array_sample)
create_interval_conformal_set.register(np.ndarray)(ArrayIntervalConformalSet.from_sample)
create_interval_conformal_set.register(ArraySample)(ArrayIntervalConformalSet.from_array_sample)
