"""Torch sample implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast, overload, override

import torch

from probly.representation.array_like import ArrayLike, ToIndices, to_numpy_array_like
from probly.representation.sample._common import Sample, SampleAxis, create_sample
from probly.representation.sample.array import ArraySample
from probly.representation.sample.axis_tracking import track_axis
from probly.representation.sample.torch_functions import TorchSampleInternals, torch_function, torch_sample_internals
from probly.representation.torch_functions import torch_average
from probly.representation.torch_like import TorchLike, TorchLikeImplementation, to_torch_like

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from types import ModuleType

    import numpy.typing as npt
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchSample[D: TorchLike | torch.Tensor](TorchLikeImplementation[D], Sample[D]):
    """A sample implementation for torch tensors."""

    tensor: D
    sample_dim: int
    weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate the sample_dim."""
        if self.sample_dim >= self.tensor.ndim:
            msg = f"sample_dim {self.sample_dim} out of bounds for tensor with ndim {self.tensor.ndim}."
            raise ValueError(msg)
        if self.sample_dim < 0:
            if self.sample_dim < -self.tensor.ndim:
                msg = f"sample_dim {self.sample_dim} out of bounds for tensor with ndim {self.tensor.ndim}."
                raise ValueError(msg)
            super(type(self), self).__setattr__("sample_dim", self.tensor.ndim + self.sample_dim)

        if not isinstance(self.tensor, TorchLikeImplementation):
            msg = "tensor must be a TorchLike object."
            raise TypeError(msg)

        if self.weights is not None and self.weights.shape != (self.sample_size,):
            msg = f"weights must have shape ({self.sample_size},), but got {self.weights.shape}."
            raise ValueError(msg)

    @override
    @classmethod
    def from_iterable(
        cls,
        samples: Iterable[ArrayLike[D]],
        weights: Iterable[float] | None = None,
        sample_dim: SampleAxis | None = None,
        sample_axis: SampleAxis | None = "auto",
        dtype: torch.dtype | None = None,
    ) -> Self:  # ty: ignore[invalid-method-override]
        """Create an TorchSample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            weights: Optional weights for the samples.
            sample_dim: The dimension along which samples are organized.
            sample_axis: Alias for sample_dim for compatibility.
            dtype: Desired data type of the array.

        Returns:
            The created TorchSample.
        """
        if sample_dim is None:
            if sample_axis is None:
                msg = "Either sample_dim or sample_axis must be not None."
                raise ValueError(msg)
            sample_dim = sample_axis
        elif sample_axis is not None and sample_axis != "auto":
            msg = "Cannot specify both sample_dim and sample_axis."
            raise ValueError(msg)

        if isinstance(samples, TorchLikeImplementation):
            if sample_dim == "auto":
                if samples.ndim == 0:
                    msg = "Cannot infer sample_dim for 0-dimensional array."
                    raise ValueError(msg)
                sample_dim = -1
            if sample_dim != 0:
                samples = torch.moveaxis(samples, 0, sample_dim)  # ty:ignore[no-matching-overload]
        else:
            samples = [to_torch_like(sample) for sample in samples]  # ty:ignore[invalid-assignment]
            if sample_dim == "auto":
                if len(samples) == 0:  # ty:ignore[invalid-argument-type]
                    msg = "Cannot infer sample_dim for empty samples."
                    raise ValueError(msg)
                sample_dim = -1
            samples = torch.stack(samples, dim=sample_dim)  # ty: ignore[invalid-argument-type]

        if dtype is not None:
            samples = samples.to(dtype=dtype)

        return cls(
            tensor=samples,  # ty:ignore[invalid-argument-type]
            sample_dim=sample_dim,
            weights=torch.as_tensor(weights, device=samples.device) if weights is not None else None,
        )

    def __len__(self) -> int:
        """Return the len of the array."""
        return len(self.tensor)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over axis 0 of the sample wrapper."""
        for index in range(len(self)):
            yield self[index]

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

    @override
    def size(self, dim: int | None = None) -> int | torch.Size:
        """The total number of elements in the underlying array."""
        return self.tensor.size(dim)  # ty:ignore[no-matching-overload]

    @override
    @property
    def mT(self) -> Self:
        """The transposed view over the last two dimensions."""
        if self.ndim < 2:
            msg = "mT requires at least 2 dimensions."
            raise ValueError(msg)

        dim0 = self.ndim - 2
        dim1 = self.ndim - 1
        transposed = torch.transpose(cast("Any", self.tensor), dim0, dim1)

        if self.sample_dim == dim0:
            sample_dim = dim1
        elif self.sample_dim == dim1:
            sample_dim = dim0
        else:
            sample_dim = self.sample_dim

        return type(self)(tensor=cast("Any", transposed), sample_dim=sample_dim)

    @override
    @property
    def mH(self) -> Self:
        """The adjoint view over the last two dimensions."""
        if self.ndim < 2:
            msg = "mH requires at least 2 dimensions."
            raise ValueError(msg)

        dim0 = self.ndim - 2
        dim1 = self.ndim - 1
        adjoint = torch.adjoint(cast("Any", self.tensor))

        if self.sample_dim == dim0:
            sample_dim = dim1
        elif self.sample_dim == dim1:
            sample_dim = dim0
        else:
            sample_dim = self.sample_dim

        return type(self)(tensor=cast("Any", adjoint), sample_dim=sample_dim)

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return self.size(self.sample_dim)

    @property
    def samples(self) -> D:
        """Return an iterator over the samples."""
        if self.sample_dim == 0:
            return self.tensor
        return torch.moveaxis(self.tensor, self.sample_dim, 0)  # ty:ignore[no-matching-overload]

    @override
    def sample_mean(self) -> D:
        """Compute the mean of the sample."""
        return torch_average(self.tensor, dim=self.sample_dim, weights=self.weights)  # ty:ignore[invalid-return-type, invalid-argument-type]

    @override
    def sample_std(self, ddof: int = 0) -> D:
        """Compute the standard deviation of the sample."""
        if self.weights is not None:
            return torch.sqrt(self.sample_var(ddof=ddof))  # ty:ignore[invalid-return-type, invalid-argument-type]

        return torch.std(self.tensor, dim=self.sample_dim, correction=ddof)  # ty:ignore[no-matching-overload]

    @override
    def sample_var(self, ddof: int = 0) -> D:
        """Compute the variance of the sample."""
        tensor = self.tensor
        weights = self.weights
        if self.weights is not None:
            if ddof != 0:
                msg = "Weighted samples do not support ddof > 0."
                raise ValueError(msg)
            average = torch_average(tensor, dim=self.sample_dim, weights=weights, keepdim=True)  # ty:ignore[invalid-argument-type]
            variance = torch_average((tensor - average) ** 2, dim=self.sample_dim, weights=weights)  # ty:ignore[unsupported-operator]
            return variance  # ty:ignore[invalid-return-type]

        return torch.var(self.tensor, dim=self.sample_dim, correction=ddof)  # ty:ignore[no-matching-overload]

    @override
    def concat(self, other: Sample[D]) -> Self:
        if isinstance(other, TorchSample):
            other_tensor = torch.moveaxis(other.tensor, other.sample_dim, self.sample_dim)  # ty:ignore[no-matching-overload]
        else:
            other_tensor = torch.stack(list(other.samples), dim=self.sample_dim)  # ty:ignore[invalid-argument-type]

        concatenated = torch.cat((self.tensor, other_tensor), dim=self.sample_dim)  # ty:ignore[no-matching-overload]

        weights = self.weights
        other_weights = other.weights if isinstance(other, TorchSample) else None

        if weights is not None or other_weights is not None:
            if weights is None:
                weights = torch.ones(self.sample_size, device=self.tensor.device)
            other_weights = (
                torch.ones(other.sample_size, device=other_tensor.device)
                if other_weights is None
                else torch.as_tensor(other_weights, device=other_tensor.device)
            )
            weights = torch.cat((weights, other_weights), dim=0)

        return type(self)(tensor=concatenated, sample_dim=self.sample_dim, weights=weights)

    def move_sample_dim(self, new_sample_dim: int) -> TorchSample:
        """Return a new TorchSample with the sample dimension moved to new_sample_dim.

        Args:
            new_sample_dim: The new sample dimension.

        Returns:
            A new TorchSample with the sample dimension moved.
        """
        moved_array = torch.moveaxis(self.tensor, self.sample_dim, new_sample_dim)  # ty:ignore[no-matching-overload]
        return type(self)(tensor=moved_array, sample_dim=new_sample_dim, weights=self.weights)

    def move_sample_axis(self, new_sample_axis: int) -> TorchSample:
        """Alias for :meth:`TorchSample.move_sample_dim`."""
        return self.move_sample_dim(new_sample_axis)

    def __getitem__(self, index: ToIndices) -> TorchLikeImplementation[D] | D:
        """Get a sample by index."""
        new_tensor = self.tensor[index]  # ty:ignore[invalid-argument-type]

        if not hasattr(new_tensor, "ndim"):
            return new_tensor

        track_result = track_axis(index, self.sample_dim, self.tensor.ndim, torch_indexing=True)

        if track_result is None:
            return new_tensor  # ty:ignore[invalid-return-type]

        weights = self.weights

        if weights is not None:
            weights_index = track_result.index
            if weights_index is NotImplemented:
                msg = "Weighted samples do not support this indexing operation."
                raise IndexError(msg)
            weights = weights[weights_index]  # ty:ignore[invalid-argument-type]

        return type(self)(tensor=new_tensor, sample_dim=track_result.new_axis, weights=weights)

    def __setitem__(self, index: ToIndices, value: object) -> None:
        """Set a sample by index."""
        cast("Any", self.tensor)[index] = value

    @override
    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        return self.tensor.__array_namespace__(api_version=api_version)  # ty:ignore[invalid-argument-type]

    def __array_like__(self, dtype: npt.DTypeLike | None = None, /, *, copy: bool | None = None) -> ArraySample[Any]:
        """Convert to a NumpyArrayLike."""
        array = to_numpy_array_like(self.tensor, dtype=dtype, copy=copy)
        return ArraySample(array=array, sample_axis=self.sample_dim)

    @override
    def numpy(self, *, force: bool = False) -> NDArray[Any]:
        """Convert to a numpy array."""
        return self.tensor.numpy(force=force)  # ty:ignore[invalid-argument-type]

    @override
    def detach(self) -> Self:
        """Return a detached copy of the sample tensor wrapper."""
        return type(self)(
            tensor=cast("Any", self.tensor).detach(),
            sample_dim=self.sample_dim,
            weights=self.weights.detach() if self.weights is not None else None,
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        """Handle torch functions."""
        del cls
        return torch_function(func, types, args, {} if kwargs is None else kwargs)

    def to(self, *args: Any, **kwargs: Any) -> Self:  # noqa: ANN401
        """Moves and/or casts the underlying tensor. See `torch.Tensor.to` for details.

        Args:
            *args: Positional arguments to pass to `torch.Tensor.to`.
            **kwargs: Keyword arguments to pass to `torch.Tensor.to`.

        Returns:
            A copy of the TorchSample.
        """
        tensor = self.tensor.to(*args, **kwargs)  # ty:ignore[no-matching-overload]
        weights = self.weights.to(*args, **kwargs) if self.weights is not None else None

        if tensor is self.tensor and weights is self.weights:
            return self

        return type(self)(tensor=tensor, sample_dim=self.sample_dim, weights=weights)

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> TorchLikeImplementation[Any]:
        """Convert to a TorchLike."""
        return self.to(dtype=dtype, device=device, copy=copy)


@torch_sample_internals.register(TorchSample)
def _[D: TorchLike](sample: TorchSample[D]) -> TorchSampleInternals[D]:
    """Get internals for a TorchSample."""
    return TorchSampleInternals[D](
        create=type(sample),
        tensor=sample.tensor,
        sample_dim=sample.sample_dim,
        weights=sample.weights,
    )


create_sample.register(
    torch.Tensor | TorchLikeImplementation,
    TorchSample.from_iterable,
)
