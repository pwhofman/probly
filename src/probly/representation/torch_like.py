"""Protocol for ndarray-like objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self, overload, runtime_checkable

import torch

from lazy_dispatch import ProtocolRegistry, lazydispatch
from probly.representation.array_like import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class TorchTensorLikeConvertible[DT](ArrayLike[DT], Protocol):
    """Protocol for array-like objects that can be converted to torch tensors."""

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> TorchTensorLike[Any]:
        """Convert to a TorchTensorLike."""


@runtime_checkable
class TorchTensorLike[DT](ArrayLike[DT], ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for array-like objects that behave like Torch tensors."""

    @property
    def mT(self) -> Self:  # noqa: N802
        """The transposed version of the underlying array."""

    @property
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


TorchTensorLike.register(torch.Tensor)


@lazydispatch
def to_torch_tensor_like[DT](
    data: ArrayLike[DT],
    dtype: torch.dtype | None = None,
    /,
    *,
    device: torch.device | str | None = None,
    copy: bool = False,
) -> TorchTensorLike[Any] | torch.Tensor:
    """Convert an ArrayLike to a TorchTensorLike.

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


@to_torch_tensor_like.register(TorchTensorLikeConvertible)
def _to_numpy_array_like_convertible[DT](
    data: TorchTensorLikeConvertible[DT],
    dtype: torch.dtype | None = None,
    /,
    *,
    device: torch.device | str | None = None,
    copy: bool = False,
) -> TorchTensorLike[Any] | torch.Tensor:
    return data.__torch_like__(dtype, device=device, copy=copy)
