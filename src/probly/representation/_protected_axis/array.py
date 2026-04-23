"""Utilities for representations with protected trailing array axes."""

from __future__ import annotations

from abc import ABC
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast, override

import numpy as np

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    normalize_axes,
    normalize_axis,
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
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {}
    permitted_functions: ClassVar[set[Callable[..., Any]]] = set()

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

        permitted_ufuncs = getattr(cls, "permitted_ufuncs", {})
        if not isinstance(permitted_ufuncs, dict):
            msg = f"{cls.__name__}.permitted_ufuncs must be a dict[np.ufunc, list[str]]."
            raise TypeError(msg)

        normalized_permitted_ufuncs: dict[np.ufunc, list[str]] = {}
        for ufunc, methods in permitted_ufuncs.items():
            if not isinstance(ufunc, np.ufunc):
                msg = f"{cls.__name__}.permitted_ufuncs keys must be numpy ufuncs."
                raise TypeError(msg)
            if not isinstance(methods, list) or not all(isinstance(method, str) for method in methods):
                msg = f"{cls.__name__}.permitted_ufuncs[{ufunc.__name__!r}] must be a list[str]."
                raise TypeError(msg)
            normalized_permitted_ufuncs[ufunc] = [cast("str", method) for method in methods]
        cls.permitted_ufuncs = normalized_permitted_ufuncs

        permitted_functions = getattr(cls, "permitted_functions", set())
        if not isinstance(permitted_functions, set) or not all(callable(func) for func in permitted_functions):
            msg = f"{cls.__name__}.permitted_functions must be a set of callables."
            raise TypeError(msg)
        cls.permitted_functions = set(permitted_functions)

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

    @override
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

    @override
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

    def __array_ufunc__(  # noqa: C901, PLR0912, PLR0915
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> object:
        permitted_methods = type(self).permitted_ufuncs.get(ufunc)
        if permitted_methods is None or method not in permitted_methods:
            return NotImplemented

        if method not in {"__call__", "reduce", "accumulate", "reduceat"}:
            return NotImplemented

        protected_axes = type(self).protected_axes
        self_values = self.protected_values()
        input_values: list[dict[str, ArrayProtectedValue] | None] = []
        for value in inputs:
            if isinstance(value, type(self)):
                value_protected_axes = type(value).protected_axes
                if value_protected_axes != protected_axes:
                    msg = "All protected inputs must share identical protected_axes definitions."
                    raise ValueError(msg)
                input_values.append(value.protected_values())
            else:
                input_values.append(None)

        out = kwargs.get("out")
        out_items = out if isinstance(out, tuple) else ((out,) if out is not None else ())
        out_values: list[dict[str, ArrayProtectedValue] | None] = []
        for value in out_items:
            if isinstance(value, type(self)):
                value_protected_axes = type(value).protected_axes
                if value_protected_axes != protected_axes:
                    msg = "All protected outputs must share identical protected_axes definitions."
                    raise ValueError(msg)
                out_values.append(value.protected_values())
            else:
                out_values.append(None)

        if out is not None and len(protected_axes) != 1 and any(values is None for values in out_values):
            msg = "non-protected out is only supported for single-field protected objects."
            raise TypeError(msg)

        def map_reduce_axis(axis: object, batch_ndim: int) -> int | tuple[int, ...]:
            if axis is None:
                return tuple(range(batch_ndim))
            if isinstance(axis, int):
                return normalize_axis(axis, batch_ndim)
            if isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
                axis_tuple = cast("tuple[int, ...]", tuple(axis))
                return normalize_axes(axis_tuple, batch_ndim)
            msg = "reduce axis must be None, an int, or a tuple/list of ints."
            raise TypeError(msg)

        def map_indexed_axis(axis: object, batch_ndim: int, *, method_name: str) -> int:
            if not isinstance(axis, int):
                msg = f"{method_name} axis must be an int."
                raise TypeError(msg)
            return normalize_axis(axis, batch_ndim)

        results: dict[str, ArrayProtectedValue] = {}
        for name, axes_count in protected_axes.items():
            field_kwargs = dict(kwargs)

            if out is not None:
                mapped_out_items = tuple(
                    field_values[name] if field_values is not None else item
                    for item, field_values in zip(out_items, out_values, strict=True)
                )
                if isinstance(out, tuple):
                    if method == "__call__":
                        field_kwargs["out"] = mapped_out_items
                    else:
                        if len(mapped_out_items) != 1:
                            msg = f"ufunc method {method!r} expects a single out value."
                            raise TypeError(msg)
                        field_kwargs["out"] = mapped_out_items[0]
                else:
                    field_kwargs["out"] = mapped_out_items[0]

            field_inputs = tuple(
                protected[name] if protected is not None else value
                for value, protected in zip(inputs, input_values, strict=True)
            )
            field_input_items = list(field_inputs)

            if method in {"reduce", "accumulate", "reduceat"}:
                batch_ndim = value_ndim(self_values[name]) - axes_count

                if method == "reduce":
                    raw_axis = field_kwargs.get("axis", field_input_items[1] if len(field_input_items) > 1 else 0)
                    field_kwargs["axis"] = map_reduce_axis(raw_axis, batch_ndim)
                    field_input_items = field_input_items[:1]
                elif method == "accumulate":
                    raw_axis = field_kwargs.get("axis", field_input_items[1] if len(field_input_items) > 1 else 0)
                    field_kwargs["axis"] = map_indexed_axis(raw_axis, batch_ndim, method_name="accumulate")
                    field_input_items = field_input_items[:1]
                else:
                    if len(field_input_items) < 2:
                        msg = "reduceat requires indices as its second argument."
                        raise TypeError(msg)
                    raw_axis = field_kwargs.get("axis", field_input_items[2] if len(field_input_items) > 2 else 0)
                    field_kwargs["axis"] = map_indexed_axis(raw_axis, batch_ndim, method_name="reduceat")
                    field_input_items = field_input_items[:2]

            result = getattr(ufunc, method)(*field_input_items, **field_kwargs)

            if out is not None:
                continue

            if axes_count == 0 and not hasattr(result, "ndim"):
                result = np.asarray(result)

            result_ndim = value_ndim(result)
            result_shape = value_shape(result)
            original_shape = value_shape(self_values[name])
            if result_ndim < axes_count:
                msg = f"Ufunc operation removed protected trailing axes for field {name!r}."
                raise ValueError(msg)
            if protected_shape(result_shape, axes_count) != protected_shape(original_shape, axes_count):
                msg = f"Ufunc operation modified protected trailing axes for field {name!r}."
                raise ValueError(msg)

            if isinstance(result, np.ndarray):
                result = self._postprocess_ufunc_result(result, ufunc=ufunc, method=method)

            results[name] = cast("ArrayProtectedValue", result)

        if out is not None:
            if method == "__call__":
                return out
            return out_items[0] if isinstance(out, tuple) else out

        expected_batch_shape: tuple[int, ...] | None = None
        for name, value in results.items():
            axes_count = protected_axes[name]
            current_batch_shape = batch_shape(value_shape(value), axes_count)
            if expected_batch_shape is None:
                expected_batch_shape = current_batch_shape
            elif current_batch_shape != expected_batch_shape:
                msg = "Ufunc operation produced inconsistent batch-shapes across protected fields."
                raise ValueError(msg)

        return self.with_protected_values(results)

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
