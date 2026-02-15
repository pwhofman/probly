"""Torch sample implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, overload, override

import numpy as np
import torch

from probly.representation.sampling.common_sample import Sample, SampleAxis, create_sample

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchTensorSample(Sample[torch.Tensor]):
    """A sample implementation for torch tensors."""

    tensor: torch.Tensor
    sample_dim: int

    def __post_init__(self) -> None:
        """Validate the sample_dim."""
        if self.sample_dim >= self.tensor.ndim:
            msg = f"sample_dim {self.sample_dim} out of bounds for tensor with ndim {self.tensor.ndim}."
            raise ValueError(msg)
        if self.sample_dim < 0:
            if self.sample_dim < -self.tensor.ndim:
                msg = f"sample_dim {self.sample_dim} out of bounds for tensor with ndim {self.tensor.ndim}."
                raise ValueError(msg)
            super().__setattr__("sample_dim", self.tensor.ndim + self.sample_dim)

        if not isinstance(self.tensor, torch.Tensor):
            msg = "tensor must be a torch tensor."
            raise TypeError(msg)

    @classmethod
    def from_iterable(  # noqa: C901
        cls,
        samples: Iterable[torch.Tensor],
        sample_dim: SampleAxis | None = None,
        sample_axis: SampleAxis | None = "auto",
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Create an TorchTensorSample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_dim: The dimension along which samples are organized.
            sample_axis: Alias for sample_dim for compatibility.
            dtype: Desired data type of the array.

        Returns:
            The created TorchTensorSample.
        """
        if sample_dim is None:
            if sample_axis is None:
                msg = "Either sample_dim or sample_axis must be not None."
                raise ValueError(msg)
            sample_dim = sample_axis
        elif sample_axis is not None and sample_axis != "auto":
            msg = "Cannot specify both sample_dim and sample_axis."
            raise ValueError(msg)

        if isinstance(samples, torch.Tensor):
            if sample_dim == "auto":
                if samples.ndim == 0:
                    msg = "Cannot infer sample_dim for 0-dimensional array."
                    raise ValueError(msg)
                sample_dim = 0 if samples.ndim == 1 else 1
            if sample_dim != 0:
                samples = torch.moveaxis(samples, 0, sample_dim)
        else:
            if not isinstance(samples, (tuple, list)):
                samples = list(samples)
            if sample_dim == "auto":
                if len(samples) == 0:
                    msg = "Cannot infer sample_dim for empty samples."
                    raise ValueError(msg)
                first_sample = samples[0]
                sample_dim = (
                    (0 if first_sample.ndim == 0 else 1) if isinstance(first_sample, (np.ndarray, torch.Tensor)) else 0
                )
            samples = torch.stack(samples, dim=sample_dim)

        if dtype is not None:
            samples = samples.to(dtype=dtype)

        return cls(tensor=samples, sample_dim=sample_dim)

    def __len__(self) -> int:
        """Return the len of the array."""
        return len(self.tensor)

    @property
    def sample_axis(self) -> int:
        """The axis along which samples are organized."""
        return self.sample_dim

    @property
    def dtype(self) -> torch.dtype:
        """The data type of the underlying array."""
        return self.tensor.dtype

    @property
    def device(self) -> Any:  # noqa: ANN401
        """The device of the underlying array."""
        return self.tensor.device

    @property
    def mT(self) -> torch.Tensor:  # noqa: N802
        """The transposed version of the underlying array."""
        return self.tensor.mT

    @property
    def ndim(self) -> int:
        """The number of dimensions of the underlying array."""
        return self.tensor.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying array."""
        return self.tensor.shape

    @overload
    def size(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = ...) -> torch.Size: ...

    def size(self, dim: int | None = None) -> torch.Size | int:
        """The total number of elements in the underlying array."""
        return self.tensor.size(dim)

    @property
    def T(self) -> torch.Tensor:  # noqa: N802
        """The transposed version of the underlying array."""
        return self.tensor.T

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return self.size(self.sample_dim)

    @property
    def samples(self) -> torch.Tensor:
        """Return an iterator over the samples."""
        if self.sample_dim == 0:
            return self.tensor
        return torch.moveaxis(self.tensor, self.sample_dim, 0)

    def sample_mean(self) -> torch.Tensor:
        """Compute the mean of the sample."""
        return self.tensor.mean(dim=self.sample_dim)

    def sample_std(self, ddof: int = 1) -> torch.Tensor:
        """Compute the standard deviation of the sample."""
        return self.tensor.std(dim=self.sample_dim, correction=ddof)

    def sample_var(self, ddof: int = 1) -> torch.Tensor:
        """Compute the variance of the sample."""
        return self.tensor.var(dim=self.sample_dim, correction=ddof)

    @override
    def concat(self, other: Sample[torch.Tensor]) -> Self:
        if isinstance(other, TorchTensorSample):
            other_tensor = torch.moveaxis(other.tensor, other.sample_dim, self.sample_dim)
        else:
            other_tensor = torch.stack(list(other.samples), dim=self.sample_dim)

        concatenated = torch.cat((self.tensor, other_tensor), dim=self.sample_dim)
        return type(self)(tensor=concatenated, sample_dim=self.sample_dim)

    def move_sample_dim(self, new_sample_dim: int) -> TorchTensorSample:
        """Return a new TorchTensorSample with the sample dimension moved to new_sample_dim.

        Args:
            new_sample_dim: The new sample dimension.

        Returns:
            A new TorchTensorSample with the sample dimension moved.
        """
        moved_array = torch.moveaxis(self.tensor, self.sample_dim, new_sample_dim)
        return type(self)(tensor=moved_array, sample_dim=new_sample_dim)

    def move_sample_axis(self, new_sample_axis: int) -> TorchTensorSample:
        """Alias for :meth:`TorchTensorSample.move_sample_dim`."""
        return self.move_sample_dim(new_sample_axis)

    def __array__(self, dtype: npt.DTypeLike = None, copy: bool | None = None) -> np.ndarray:
        """Get the underlying numpy array.

        Args:
            dtype: Desired data type of the array.
            copy: Whether to return a copy of the array.

        Returns:
            The underlying numpy array.
        """
        return np.asarray(self.tensor, dtype=dtype, copy=copy)

    def to(self, *args: Any, **kwargs: Any) -> Self:  # noqa: ANN401
        """Moves and/or casts the underlying tensor. See `torch.Tensor.to` for details.

        Args:
            *args: Positional arguments to pass to `torch.Tensor.to`.
            **kwargs: Keyword arguments to pass to `torch.Tensor.to`.

        Returns:
            A copy of the TorchTensorSample.
        """
        return type(self)(tensor=self.tensor.to(*args, **kwargs), sample_dim=self.sample_dim)


create_sample.register(torch.Tensor, TorchTensorSample.from_iterable)
