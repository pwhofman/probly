"""Utilities for representations with protected trailing tensor axes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from operator import attrgetter
from sys import modules
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self, get_args, get_origin, overload, override

import numpy as np
import torch

from probly.representation._protected_axis.torch_functions import torch_function
from probly.representation.torch_like import TorchTensorLike

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import numpy.typing as npt

    from probly.representation.array_like import ToIndices


def _is_torch_tensor_annotation(annotation: object) -> bool:
    origin = get_origin(annotation)

    if annotation is torch.Tensor or origin is torch.Tensor:
        return True

    if origin is None:
        return False

    if origin is Annotated:
        args = get_args(annotation)
        return len(args) > 0 and _is_torch_tensor_annotation(args[0])

    return False


class TorchAxisProtected[T](TorchTensorLike[T], ABC):
    """ABC for tensor-backed representations with protected trailing axes."""

    protected_axes: ClassVar[int]
    axis_protected_tensor_name: ClassVar[str]
    axis_protected_tensor_getter: ClassVar[Callable[[Any], torch.Tensor]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Resolve and cache the protected tensor attribute for subclasses."""
        super().__init_subclass__(**kwargs)

        if cls is TorchAxisProtected:
            return

        protected_axes = getattr(cls, "protected_axes", None)
        if not isinstance(protected_axes, int) or protected_axes < 1:
            msg = f"{cls.__name__} must define protected_axes as an int >= 1."
            raise TypeError(msg)

        declared_annotations = cls.__dict__.get("__annotations__", {})
        module_globals = vars(modules[cls.__module__])

        for name in declared_annotations:
            annotation = declared_annotations[name]
            if isinstance(annotation, str):
                annotation = eval(annotation, module_globals, vars(cls))  # noqa: S307

            if _is_torch_tensor_annotation(annotation):
                cls.axis_protected_tensor_name = name
                cls.axis_protected_tensor_getter = attrgetter(name)
                return

        msg = f"{cls.__name__} must declare at least one annotated torch.Tensor attribute."
        raise TypeError(msg)

    def protected_tensor(self) -> torch.Tensor:
        """Return the tensor carrying the protected trailing axes."""
        return self.__class__.axis_protected_tensor_getter(self)

    @abstractmethod
    def with_protected_tensor(self, tensor: torch.Tensor) -> Self:
        """Create a new object with a replaced protected tensor."""

    @override
    def __len__(self) -> int:
        """Return the length along the first batch dimension."""
        if self.ndim == 0:
            msg = "len() of unsized distribution"
            raise TypeError(msg)
        return len(self.protected_tensor())

    @override
    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """Get the array namespace of the underlying tensor."""
        return self.protected_tensor().__array_namespace__(api_version=api_version)  # ty:ignore[unresolved-attribute]

    @override
    @property
    def dtype(self) -> torch.dtype:
        """The data type of the underlying tensor."""
        return self.protected_tensor().dtype

    @override
    @property
    def device(self) -> torch.device:
        """The device of the underlying tensor."""
        return self.protected_tensor().device

    @override
    @property
    def ndim(self) -> int:
        """Number of batch dimensions (excluding protected trailing axes)."""
        return self.protected_tensor().ndim - self.protected_axes

    @override
    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape (excluding protected trailing axes)."""
        return tuple(self.protected_tensor().shape[: -self.protected_axes])

    @property
    def protected_shape(self) -> tuple[int, ...]:
        """Shape of the protected trailing axes."""
        return tuple(self.protected_tensor().shape[-self.protected_axes :])

    @overload
    def size(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = ...) -> torch.Size: ...

    @override
    def size(self, dim: int | None = None) -> int | torch.Size:
        """Return the size in batch-space semantics."""
        if dim is None:
            return torch.Size(self.shape)

        normalized_dim = dim + self.ndim if dim < 0 else dim
        if normalized_dim < 0 or normalized_dim >= self.ndim:
            msg = f"dim {dim} out of bounds for batch dimensions with ndim {self.ndim}."
            raise IndexError(msg)

        return self.shape[normalized_dim]

    @override
    @property
    def mT(self) -> Self:
        """Matrix transpose in batch-space semantics."""
        if self.ndim < 2:
            msg = "mT requires at least 2 batch dimensions."
            raise ValueError(msg)

        batch_axes = list(range(self.ndim))
        batch_axes[-2], batch_axes[-1] = batch_axes[-1], batch_axes[-2]
        full_axes = (*batch_axes, *range(self.ndim, self.ndim + self.protected_axes))

        return self.with_protected_tensor(torch.permute(self.protected_tensor(), full_axes))

    @override
    @property
    def mH(self) -> Self:
        """Adjoint transpose in batch-space semantics."""
        transposed = self.mT.protected_tensor()
        if torch.is_complex(transposed):
            transposed = torch.conj(transposed)

        return self.with_protected_tensor(transposed)

    def _validate_result_preserves_protected_axes(self, result: torch.Tensor) -> None:
        if result.ndim < self.protected_axes:
            msg = "Operation removed protected trailing axes."
            raise ValueError(msg)
        if tuple(result.shape[-self.protected_axes :]) != self.protected_shape:
            msg = "Operation modified protected trailing axes."
            raise ValueError(msg)

    def _index_with_protected_axes(self, index: ToIndices) -> tuple[Any, ...]:
        index_tuple = index if isinstance(index, tuple) else (index,)
        return (*index_tuple, *(slice(None),) * self.protected_axes)

    def _coerce_assignment_value(self, value: object) -> torch.Tensor:
        value_tensor = (
            value.protected_tensor() if isinstance(value, type(self)) else torch.as_tensor(value, device=self.device)
        )

        if value_tensor.ndim < self.protected_axes:
            msg = "Assigned value must include all protected trailing axes."
            raise ValueError(msg)

        if tuple(value_tensor.shape[-self.protected_axes :]) != self.protected_shape:
            msg = "Assigned value modifies protected trailing axes."
            raise ValueError(msg)

        return value_tensor

    @override
    def __getitem__(self, index: ToIndices, /) -> Self | T:
        """Return a subset while preserving protected trailing axes."""
        full_index = self._index_with_protected_axes(index)
        result = self.protected_tensor()[full_index]

        if not isinstance(result, torch.Tensor):
            msg = "Indexing cannot remove protected trailing axes."
            raise IndexError(msg)

        self._validate_result_preserves_protected_axes(result)
        return self.with_protected_tensor(result)

    @override
    def __setitem__(self, index: ToIndices, value: object, /) -> None:
        """Set a subset while preserving protected trailing axes."""
        full_index = self._index_with_protected_axes(index)

        target = self.protected_tensor()[full_index]
        if not isinstance(target, torch.Tensor):
            msg = "Indexing cannot remove protected trailing axes."
            raise IndexError(msg)

        self._validate_result_preserves_protected_axes(target)
        self.protected_tensor()[full_index] = self._coerce_assignment_value(value)

    @override
    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move and/or cast the underlying tensor."""
        tensor = self.protected_tensor().to(*args, **kwargs)
        if tensor is self.protected_tensor():
            return self

        return self.with_protected_tensor(tensor)

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> TorchTensorLike[Any]:
        """Convert to a TorchTensorLike."""
        return self.to(dtype=dtype, device=device, copy=copy)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        """Handle torch functions with protected-axis semantics."""
        del cls
        return torch_function(func, types, args, {} if kwargs is None else kwargs)

    @overload
    def __array__(self) -> npt.NDArray[Any]: ...

    @overload
    def __array__(self, dtype: npt.DTypeLike) -> npt.NDArray[Any]: ...

    @override
    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[Any]:
        """Convert to a numpy array for interoperability."""
        return np.asarray(self.protected_tensor().detach().cpu().numpy(), dtype=dtype, copy=copy)
