"""Classes representing credal sets."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, override

import numpy as np

from probly.representation.credal_set.common import CategoricalCredalSet, DiscreteCredalSet
from probly.representation.sampling.sample import ArraySample

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from probly.representation.sampling.common_sample import Sample


class ArrayCategoricalCredalSet(CategoricalCredalSet[np.ndarray], metaclass=ABCMeta):
    """A credal set of predictions stored in a numpy array."""

    @override
    @classmethod
    def from_sample(cls, sample: Sample[np.ndarray], distribution_axis: int = -1) -> Self:
        array_sample = ArraySample.from_sample(sample)
        return cls.from_array_sample(array_sample, distribution_axis=distribution_axis)

    @classmethod
    @abstractmethod
    def from_array_sample(
        cls,
        sample: ArraySample[np.ndarray],
        distribution_axis: int = -1,
    ) -> Self:
        """Create a credal set from an ArraySample.

        Args:
            sample: The sample to create the credal set from.
            distribution_axis: The axis in each sample containing the categorical probabilities.

        Returns:
            The created credal set.
        """
        msg = "from_array_sample method not implemented."
        raise NotImplementedError(msg)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayDiscreteCredalSet(ArrayCategoricalCredalSet, DiscreteCredalSet[np.ndarray]):
    """A discrete credal set over a finite set of distributions stored in a numpy array.

    Internall, a discrete credal set is represented as a numpy array of shape
    (..., num_members, num_classes)
    """

    array: np.ndarray

    @override
    @classmethod
    def from_array_sample(
        cls,
        sample: ArraySample[np.ndarray],
        distribution_axis: int = -1,
    ) -> Self:
        if distribution_axis < 0:
            distribution_axis += sample.ndim - 1

        array = np.moveaxis(sample.samples, (0, distribution_axis + 1), (-2, -1))

        return cls(array=array)

    def __array_namespace__(self) -> Any:  # noqa: ANN401
        """Get the array namespace of the underlying array."""
        return self.array.__array_namespace__()

    @property
    def device(self) -> str:
        """Return the device of the credal set array."""
        return self.array.device

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the credal set array."""
        return self.array.dtype  # type: ignore[no-any-return]

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the credal set array."""
        return self.array.ndim - 2

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the credal set array."""
        return self.array.shape[:-2]  # type: ignore[no-any-return]

    def __len__(self) -> int:
        """Return the number of members in the credal set."""
        shape = self.shape

        if len(shape) == 0:
            msg = "len() of unsized credal set"
            raise TypeError(msg)

        return shape[0]

    def __array__(self, dtype: DTypeLike = None, copy: bool | None = None) -> np.ndarray:
        """Get the underlying numpy array.

        Args:
            dtype: Desired data type of the array.
            copy: Whether to return a copy of the array.

        Returns:
            The underlying numpy array.
        """
        if dtype is None and not copy:
            return self.array

        return np.asarray(self.array, dtype=dtype, copy=copy)

    def lower(self) -> np.ndarray:
        """Compute the lower envelope of the credal set."""
        return np.min(self.array, axis=-2)  # type: ignore[no-any-return]

    def upper(self) -> np.ndarray:
        """Compute the upper envelope of the credal set."""
        return np.max(self.array, axis=-2)  # type: ignore[no-any-return]

    def copy(self) -> Self:
        """Create a copy of the ArraySample.

        Returns:
            A copy of the ArraySample.
        """
        return type(self)(array=self.array.copy())

    def to_device(self, device: Literal["cpu"]) -> Self:
        """Move the underlying array to the specified device.

        Args:
            device: The target device.

        Returns:
            A new ArrayDiscreteCredalSet on the specified device.
        """
        if device == self.device:
            return self

        return type(self)(array=self.array.to_device(device))

    def __eq__(self, value: Any) -> Self:  # type: ignore[override]  # noqa: ANN401, PYI032
        """Vectorized equality comparison."""
        return np.equal(self, value)  # type: ignore[return-value]

    def __hash__(self) -> int:
        """Compute the hash of the ArraySample."""
        return super().__hash__()
