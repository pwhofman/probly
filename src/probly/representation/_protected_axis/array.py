"""Utilities for representations with protected trailing array axes."""

from __future__ import annotations

from abc import ABC
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast, override

import numpy as np

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    protected_shape,
    value_ndim,
    value_shape,
)
from probly.representation._protected_axis.array_functions import array_function
from probly.representation.array_like import ArrayFlagsLike, NumpyArrayLike, NumpyArrayLikeImplementation, ToIndices

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from numpy.typing import DTypeLike


type ArrayProtectedValue = NumpyArrayLike[Any] | np.ndarray


def _validate_field_ndim(name: str, ndim: int, protected_axes: int) -> None:
    if ndim < protected_axes:
        msg = f"Protected field {name!r} has ndim {ndim}, expected >= {protected_axes}."
        raise ValueError(msg)


class ArrayAxisProtected[T: NumpyArrayLike | np.ndarray](NumpyArrayLikeImplementation[T], ABC):
    """ABC for representations with one or multiple protected array-like fields."""

    protected_axes: ClassVar[dict[str, int]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if cls is ArrayAxisProtected:
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

    @classmethod
    def primary_protected_name(cls) -> str:
        """Return the first protected field (dict order)."""
        return next(iter(cls.protected_axes))

    def protected_values(self) -> dict[str, ArrayProtectedValue]:
        """Return all protected field values.

        The values are preserved as-is and are not coerced to ``np.ndarray``.
        """
        values: dict[str, ArrayProtectedValue] = {}
        primary_name = type(self).primary_protected_name()
        primary_batch: tuple[int, ...] | None = None

        for name, axes in type(self).protected_axes.items():
            value = cast("ArrayProtectedValue", getattr(self, name))
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

    def protected_value(self) -> ArrayProtectedValue:
        """Return the primary protected value."""
        primary_name = type(self).primary_protected_name()
        return self.protected_values()[primary_name]

    def with_protected_values(self, values: dict[str, ArrayProtectedValue]) -> Self:
        """Return a copy with updated protected field values."""
        return replace(self, **values)  # ty:ignore[invalid-argument-type]

    def with_protected_value(self, value: ArrayProtectedValue) -> Self:
        """Return a copy with a replaced primary protected value."""
        if len(type(self).protected_axes) != 1:
            msg = "with_protected_value is only supported for single-field protected objects."
            raise TypeError(msg)
        return self.with_protected_values({type(self).primary_protected_name(): value})

    @override
    def __len__(self) -> int:
        if self.ndim == 0:
            msg = "len() of unsized representation"
            raise TypeError(msg)
        return len(cast("Any", self.protected_value()))

    @override
    def __array_namespace__(
        self, /, *, api_version: Literal["2022.12", "2023.12", "2024.12"] | None = None
    ) -> ModuleType:
        return self.protected_value().__array_namespace__(api_version=api_version)

    @override
    def __array__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> np.ndarray:
        if len(type(self).protected_axes) != 1:
            msg = "Cannot convert multi-field protected object to a single numpy array."
            raise TypeError(msg)
        return np.asarray(self.protected_value(), dtype=dtype, copy=copy)

    @override
    @property
    def dtype(self) -> np.dtype:
        return self.protected_value().dtype

    @override
    @property
    def device(self) -> str:
        return self.protected_value().device

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

    @override
    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @override
    @property
    def flags(self) -> ArrayFlagsLike:
        return self.protected_value().flags

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

    def __getitem__(self, index: ToIndices, /) -> Self:
        values = self.protected_values()
        indexed: dict[str, ArrayProtectedValue] = {}

        for name, axes in type(self).protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            result = cast("Any", values[name])[full_index]

            if axes == 0 and not hasattr(result, "ndim"):
                result = np.asarray(result)

            result_ndim = value_ndim(result)
            result_shape = value_shape(result)
            _validate_field_ndim(name, result_ndim, axes)

            original_shape = value_shape(values[name])
            if protected_shape(result_shape, axes) != protected_shape(original_shape, axes):
                msg = f"Indexing field {name!r} modified protected trailing axes."
                raise IndexError(msg)

            indexed[name] = result

        return self.with_protected_values(indexed)

    def __setitem__(self, index: ToIndices, value: object, /) -> None:
        values = self.protected_values()
        candidate_values = self._coerce_assignment_value(value)

        for name, axes in type(self).protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            values[name][full_index] = candidate_values[name]  # ty:ignore[invalid-assignment]

    def _postprocess_ufunc_result(
        self,
        result: np.ndarray,
        *,
        ufunc: np.ufunc,
        method: str,
    ) -> np.ndarray:
        """Optionally postprocess wrapped ufunc results."""
        del ufunc, method
        return result

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> object:
        if len(type(self).protected_axes) != 1:
            msg = "__array_ufunc__ is undefined for multi-field protected objects by default."
            raise TypeError(msg)

        if method != "__call__":
            return NotImplemented

        out = kwargs.get("out")
        if out is not None:
            outs = out if isinstance(out, tuple) else (out,)
            kwargs["out"] = tuple(o.protected_value() if isinstance(o, type(self)) else o for o in outs)

        converted_inputs = tuple(x.protected_value() if isinstance(x, type(self)) else x for x in inputs)
        result = getattr(ufunc, method)(*converted_inputs, **kwargs)

        if out is not None:
            return out

        if not hasattr(result, "ndim"):
            return result

        primary_name = type(self).primary_protected_name()
        axes = type(self).protected_axes[primary_name]
        result_ndim = value_ndim(result)
        result_shape = value_shape(result)
        if result_ndim < axes:
            return result
        if protected_shape(result_shape, axes) != self.protected_shape:
            return result

        if isinstance(result, np.ndarray):
            result = self._postprocess_ufunc_result(result, ufunc=ufunc, method=method)

        return self.with_protected_value(result)

    def __array_function__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        return array_function(func, types, args, kwargs)

    def reshape(self, *shape: int | tuple[int, ...], order: str = "C", copy: bool | None = None) -> Self:
        """Return a copy with reshaped protected values."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return np.reshape(self, shape, order=order, copy=copy)  # ty:ignore[no-matching-overload]
