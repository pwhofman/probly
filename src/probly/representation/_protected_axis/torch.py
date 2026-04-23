"""Utilities for representations with protected trailing tensor axes."""

from __future__ import annotations

from abc import ABC
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload, override

import torch

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    protected_shape,
    value_ndim,
    value_shape,
)
from probly.representation._protected_axis.torch_functions import torch_function
from probly.representation.torch_like import TorchLike, TorchLikeImplementation

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import numpy as np
    from numpy.typing import DTypeLike, NDArray

    from probly.representation.array_like import ToIndices


type TorchProtectedValue = TorchLike[Any] | torch.Tensor


def _validate_field_ndim(name: str, ndim: int, protected_axes: int) -> None:
    if ndim < protected_axes:
        msg = f"Protected field {name!r} has ndim {ndim}, expected >= {protected_axes}."
        raise ValueError(msg)


class TorchAxisProtected[T: TorchLike | torch.Tensor](TorchLikeImplementation[T], ABC):
    """ABC for representations with one or multiple protected tensor-like fields."""

    protected_axes: ClassVar[dict[str, int]] = {}
    permitted_functions: ClassVar[set[Callable[..., Any]]] = set()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if cls is TorchAxisProtected:
            return

        axes = getattr(cls, "protected_axes", None)
        if not isinstance(axes, dict) or len(axes) == 0:
            msg = f"{cls.__name__} must define protected_axes as a non-empty dict[str, int]."
            raise TypeError(msg)

        for name, value in axes.items():
            if not isinstance(name, str) or len(name) == 0:
                msg = f"{cls.__name__}.protected_axes must use non-empty string keys."
                raise TypeError(msg)
            if not isinstance(value, int) or value < 0:
                msg = f"{cls.__name__}.protected_axes[{name!r}] must be an int >= 0."
                raise TypeError(msg)
            if not hasattr(cls, "__annotations__") or name not in cls.__annotations__:
                msg = f"{cls.__name__}.protected_axes refers to unknown field {name!r}."
                raise TypeError(msg)

        permitted_functions = getattr(cls, "permitted_functions", set())
        if not isinstance(permitted_functions, set) or not all(callable(func) for func in permitted_functions):
            msg = f"{cls.__name__}.permitted_functions must be a set of callables."
            raise TypeError(msg)
        cls.permitted_functions = set(permitted_functions)

    @classmethod
    def primary_protected_name(cls) -> str:
        """Return the first protected field (dict order)."""
        return next(iter(cls.protected_axes))

    def protected_values(self) -> dict[str, TorchProtectedValue]:
        """Return all protected field values as-is."""
        values: dict[str, TorchProtectedValue] = {}
        primary_name = type(self).primary_protected_name()
        primary_batch: tuple[int, ...] | None = None

        for name, axes in type(self).protected_axes.items():
            value = cast("TorchProtectedValue", getattr(self, name))
            ndim = value_ndim(value)
            shape = value_shape(value)
            _validate_field_ndim(name, ndim, axes)

            current_batch = batch_shape(shape, axes)
            if name == primary_name:
                primary_batch = current_batch
            elif primary_batch is not None and current_batch != primary_batch:
                msg = "Protected fields do not share the same batch-shape."
                raise ValueError(msg)

            values[name] = value

        return values

    def protected_value(self) -> TorchProtectedValue:
        """Return the primary protected value."""
        primary_name = type(self).primary_protected_name()
        return self.protected_values()[primary_name]

    def with_protected_values(self, values: dict[str, TorchProtectedValue]) -> Self:
        """Return a copy with updated protected field values."""
        current_values = self.protected_values()
        updates: dict[str, object] = {}

        for name in type(self).protected_axes:
            updates[name] = values.get(name, current_values[name])

        return cast("Self", replace(cast("Any", self), **updates))

    def with_protected_value(self, value: TorchProtectedValue) -> Self:
        """Return a copy with a replaced primary protected value."""
        if len(type(self).protected_axes) != 1:
            msg = "with_protected_value is only supported for single-field protected objects."
            raise TypeError(msg)
        return self.with_protected_values({type(self).primary_protected_name(): value})

    @override
    def __len__(self) -> int:
        if self.ndim == 0:
            msg = "len() of unsized distribution"
            raise TypeError(msg)
        return len(cast("Any", self.protected_value()))

    @override
    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        return cast("Any", self.protected_value()).__array_namespace__(api_version=api_version)

    @override
    @property
    def dtype(self) -> torch.dtype:
        return cast("torch.dtype", self.protected_value().dtype)

    @override
    @property
    def device(self) -> torch.device:
        return cast("torch.device", self.protected_value().device)

    @override
    @property
    def ndim(self) -> int:
        primary_name = type(self).primary_protected_name()
        axes = type(self).protected_axes[primary_name]
        return len(batch_shape(value_shape(self.protected_value()), axes))

    @override
    @property
    def shape(self) -> tuple[int, ...]:
        primary_name = type(self).primary_protected_name()
        axes = type(self).protected_axes[primary_name]
        return batch_shape(value_shape(self.protected_value()), axes)

    @property
    def protected_shape(self) -> tuple[int, ...]:
        """Protected trailing shape of the primary field."""
        primary_name = type(self).primary_protected_name()
        axes = type(self).protected_axes[primary_name]
        return protected_shape(value_shape(self.protected_value()), axes)

    @overload
    def size(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = ...) -> torch.Size: ...

    @override
    def size(self, dim: int | None = None) -> int | torch.Size:
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
        if self.ndim < 2:
            msg = "mT requires at least 2 batch dimensions."
            raise ValueError(msg)
        return cast("Self", torch.transpose(cast("Any", self), -2, -1))

    @override
    @property
    def mH(self) -> Self:
        if self.ndim < 2:
            msg = "mH requires at least 2 batch dimensions."
            raise ValueError(msg)
        return cast("Self", torch.adjoint(cast("Any", self)))

    def _index_with_protected_axes(self, index: ToIndices, protected_axes_count: int) -> tuple[Any, ...]:
        index_tuple = index if isinstance(index, tuple) else (index,)
        return (*index_tuple, *(slice(None),) * protected_axes_count)

    def _coerce_assignment_value(self, value: object) -> dict[str, object]:
        field_names = tuple(type(self).protected_axes.keys())

        if isinstance(value, type(self)):
            candidate_values: dict[str, Any] = dict(value.protected_values())
        elif isinstance(value, tuple):
            if len(value) != len(field_names):
                msg = f"Expected tuple with {len(field_names)} values for assignment."
                raise TypeError(msg)
            candidate_values = dict(zip(field_names, value, strict=True))
        elif len(field_names) == 1:
            candidate_values = {field_names[0]: value}
        else:
            msg = "Assignment to multi-field protected object requires matching instance or value tuple."
            raise TypeError(msg)

        current_values = self.protected_values()
        for name, axes in type(self).protected_axes.items():
            candidate = candidate_values[name]
            if axes == 0:
                continue

            candidate_ndim = value_ndim(candidate)
            candidate_shape = value_shape(candidate)
            _validate_field_ndim(name, candidate_ndim, axes)

            current_shape = value_shape(current_values[name])
            if protected_shape(candidate_shape, axes) != protected_shape(current_shape, axes):
                msg = f"Assigned value for field {name!r} modifies protected trailing axes."
                raise ValueError(msg)

        return candidate_values

    @override
    def __getitem__(self, index: ToIndices, /) -> Self | T:
        values = self.protected_values()
        indexed: dict[str, TorchProtectedValue] = {}

        for name, axes in type(self).protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            result = cast("Any", values[name])[full_index]

            if axes == 0 and not hasattr(result, "ndim"):
                result = torch.as_tensor(result)

            result_ndim = value_ndim(result)
            result_shape = value_shape(result)
            _validate_field_ndim(name, result_ndim, axes)

            original_shape = value_shape(values[name])
            if protected_shape(result_shape, axes) != protected_shape(original_shape, axes):
                msg = f"Indexing field {name!r} modified protected trailing axes."
                raise IndexError(msg)

            indexed[name] = cast("TorchProtectedValue", result)

        return self.with_protected_values(indexed)

    @override
    def __setitem__(self, index: ToIndices, value: object, /) -> None:
        values = self.protected_values()
        candidate_values = self._coerce_assignment_value(value)

        for name, axes in type(self).protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            cast("Any", values[name])[full_index] = candidate_values[name]

    @override
    def to(self, *args: Any, **kwargs: Any) -> Self:
        values = self.protected_values()
        updates: dict[str, TorchProtectedValue] = {}
        changed = False

        for name, value in values.items():
            converted = cast("Any", value).to(*args, **kwargs)
            updates[name] = cast("TorchProtectedValue", converted)
            changed = changed or converted is not value

        if not changed:
            return self

        return self.with_protected_values(updates)

    def __torch_like__(
        self,
        dtype: torch.dtype | None = None,
        /,
        *,
        device: torch.device | str | None = None,
        copy: bool = False,
    ) -> TorchLikeImplementation[Any]:
        return self.to(dtype=dtype, device=device, copy=copy)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        del cls
        return torch_function(func, types, args, {} if kwargs is None else kwargs)

    @override
    def numpy(self, *, force: bool = False) -> np.ndarray:
        if len(type(self).protected_axes) != 1:
            msg = "Cannot convert multi-field protected object to a single numpy array."
            raise TypeError(msg)
        return cast("Any", self.protected_value()).numpy(force=force)

    def __array__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> NDArray[Any]:  # ty: ignore[invalid-method-override]
        array = self.numpy(force=dtype is not None or bool(copy))
        if dtype is not None:
            return array.astype(dtype, copy=bool(copy))
        if copy:
            return array.copy()
        return array

    @override
    def detach(self) -> Self:
        values = {
            name: cast("TorchProtectedValue", cast("Any", value).detach())
            for name, value in self.protected_values().items()
        }
        return self.with_protected_values(values)

    def reshape(self, *shape: int | tuple[int, ...]) -> Self:
        """Return a copy with reshaped protected values."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return torch.reshape(self, shape)  # ty:ignore[invalid-return-type, invalid-argument-type]
