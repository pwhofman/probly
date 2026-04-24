"""Protocol for ndarray-like objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, Self, overload, override, runtime_checkable

import torch
from torch.overrides import handle_torch_function, has_torch_function_unary

from flextype import flexdispatch
from probly.representation.array_like import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike, NDArray


@runtime_checkable
class TorchLikeConvertible[DT](ArrayLike[DT], Protocol):
    """Protocol for array-like objects that can be converted to torch tensors."""

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> TorchLike[Any]:
        """Convert to a TorchLike."""


@runtime_checkable
class TorchLike[DT](ArrayLike[DT], Protocol):
    """Protocol for array-like objects that implement torch-specific APIs."""

    @property
    def mT(self) -> Self:  # noqa: N802
        """The transposed version of the underlying array."""

    @property
    def mH(self) -> Self:  # noqa: N802
        """The adjoint (conjugate) transposed version of the underlying array."""

    @overload
    def size(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = ...) -> torch.Size: ...

    def size(self, dim: int | None = None) -> int | torch.Size:
        """Return the size of the array along the given dimension."""

    @overload
    def to(
        self,
        dtype: torch.dtype,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self: ...

    @overload
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self: ...

    @overload
    def to(
        self,
        other: torch.Tensor,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move and/or cast the tensor, mirroring ``torch.Tensor.to``."""

    def __torch_function__(
        self, func: Callable, types: tuple[type[Any], ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:  # noqa: ANN401
        """Handle torch functions."""

    def numpy(self, *, force: bool = False) -> NDArray[Any]:
        """Convert to a numpy array."""

    def detach(self) -> Self:
        """Return a detached version of the array."""


class TorchLikeImplementation[DT](ArrayLike[DT], ABC):
    """ABC implementation for array-like objects that behave like torch tensors."""

    @property
    @abstractmethod
    def mT(self) -> Self:  # noqa: N802
        """The transposed version of the underlying array."""

    @property
    @abstractmethod
    def mH(self) -> Self:  # noqa: N802
        """The adjoint (conjugate) transposed version of the underlying array."""

    @property
    def T(self) -> Self:  # noqa: N802
        """Inverts the order of the dimensions of the underlying array."""
        return self.permute(*reversed(range(self.ndim)))  # type: ignore[no-any-return]

    @overload
    def size(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = ...) -> torch.Size: ...

    @abstractmethod
    def size(self, dim: int | None = None) -> int | torch.Size:
        """Return the size of the array along the given dimension."""

    @overload
    def to(
        self,
        dtype: torch.dtype,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self: ...

    @overload
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self: ...

    @overload
    def to(
        self,
        other: torch.Tensor,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...

    @abstractmethod
    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move and/or cast the tensor, mirroring ``torch.Tensor.to``."""

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> Self:
        """Move the array to the CPU."""
        return self.to("cpu", memory_format=memory_format)

    def cuda(
        self,
        device: torch.device | str | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self:
        """Move the array to the GPU."""
        if device is None:
            device = "cuda"
        return self.to(
            device=device,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    def clone(self, *, memory_format: torch.memory_format = torch.preserve_format) -> Self:
        """Return a copy of the array."""
        return torch.clone(self, memory_format=memory_format)  # ty:ignore[invalid-return-type, invalid-argument-type]

    def transpose(self, dim0: int, dim1: int) -> Self:
        """Return a transposed version of the array."""
        return torch.transpose(self, dim0, dim1)  # ty:ignore[no-matching-overload]

    def permute(self, *dims: torch.Size | int | tuple[int] | list[int]) -> Self:
        """Return a permuted version of the array."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]  # ty:ignore[invalid-assignment]

        return torch.permute(self, dims)  # ty:ignore[invalid-return-type, invalid-argument-type]

    @abstractmethod
    def __torch_function__(
        self, func: Callable, types: tuple[type[Any], ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:  # noqa: ANN401
        """Handle torch functions.

        Args:
            func: The torch function to apply.
            types: The types of the input arguments.
            args: The input arguments.
            kwargs: Additional keyword arguments.

        Returns:
            The result of applying the torch function.
        """

    @abstractmethod
    def numpy(self, *, force: bool = False) -> NDArray[Any]:
        """Convert to a numpy array."""

    @overload
    def __array__(self) -> NDArray[Any]: ...

    @overload
    def __array__(self, dtype: DTypeLike) -> NDArray[Any]: ...

    @override
    def __array__(  # ty: ignore[invalid-method-override]
        self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None
    ) -> NDArray[Any]:
        """Convert to a numpy array."""
        if has_torch_function_unary(self):
            return handle_torch_function(torch.Tensor.__array__, (type(self),), self, dtype=dtype)  # type: ignore[no-any-return]
        if dtype is None:
            return self.numpy()
        return self.numpy(force=True).astype(dtype, copy=copy)  # ty:ignore[no-matching-overload]

    @abstractmethod
    def detach(self) -> Self:
        """Return a detached version of the array."""

    def resolve_conj(self) -> Self:
        """Return a version of the array with any conjugate operations resolved."""
        return torch.resolve_conj(self)  # ty:ignore[invalid-return-type, invalid-argument-type]

    def resolve_neg(self) -> Self:
        """Return a version of the array with any negation operations resolved."""
        return torch.resolve_neg(self)  # ty:ignore[invalid-return-type, invalid-argument-type]


TorchLikeImplementation.register(torch.Tensor)


@flexdispatch
def to_torch_like[DT](
    data: object,
    /,
    dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | None = None,
    copy: bool = False,
) -> TorchLike[Any] | torch.Tensor:
    """Convert an ArrayLike to a TorchLike.

    If possible, use the __torch_like__ method to convert the array, otherwise use torch.as_tensor.

    Args:
        data: The data to convert.
        dtype: The desired data type of the output tensor.
        device: The desired device of the output tensor.
        copy: Whether to return a copy of the input data.

    Returns:
        The converted tensor.
    """
    res = torch.as_tensor(data, dtype=dtype, device=device)

    if res is data and copy:
        res = res.clone()

    return res


@to_torch_like.register(TorchLikeConvertible)
def _[DT](
    data: TorchLikeConvertible[DT],
    /,
    dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | None = None,
    copy: bool = False,
) -> TorchLike[Any] | torch.Tensor:
    return data.__torch_like__(dtype, device=device, copy=copy)
