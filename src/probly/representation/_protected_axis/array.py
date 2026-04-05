"""Utilities for representations with protected trailing array axes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from operator import attrgetter
from sys import modules
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self, get_args, get_origin, override

import numpy as np

from probly.representation._protected_axis.array_functions import array_function
from probly.representation.array_like import ArrayFlagsLike, NumpyArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from probly.representation.array_like import ToIndices


def _is_ndarray_annotation(annotation: object) -> bool:
    origin = get_origin(annotation)

    if annotation is np.ndarray or origin is np.ndarray:
        return True

    if origin is None:
        return False

    if origin is Annotated:
        args = get_args(annotation)
        return len(args) > 0 and _is_ndarray_annotation(args[0])

    return False


class ArrayAxisProtected[T](NumpyArrayLike[T], ABC):
    """ABC for array-backed representations with protected trailing axes."""

    protected_axes: ClassVar[int]
    axis_protected_array_name: ClassVar[str]
    axis_protected_array_getter: ClassVar[Callable[[Any], np.ndarray]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Resolve and cache the protected ndarray attribute for subclasses."""
        super().__init_subclass__(**kwargs)

        if cls is ArrayAxisProtected:
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

            if _is_ndarray_annotation(annotation):
                cls.axis_protected_array_name = name
                cls.axis_protected_array_getter = attrgetter(name)
                return

        msg = f"{cls.__name__} must declare at least one annotated np.ndarray attribute."
        raise TypeError(msg)

    def protected_array(self) -> np.ndarray:
        """Return the ndarray carrying the protected trailing axes."""
        return self.__class__.axis_protected_array_getter(self)

    @abstractmethod
    def with_protected_array(self, array: np.ndarray) -> Self:
        """Create a new object with a replaced protected ndarray."""

    @override
    def __len__(self) -> int:
        """Return the length along the first dimension."""
        if self.ndim == 0:
            msg = "len() of unsized distribution"
            raise TypeError(msg)
        return len(self.protected_array())

    @override
    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """Get the array namespace of the underlying array."""
        return self.protected_array().__array_namespace__(api_version=api_version)  # ty:ignore[invalid-argument-type]

    @override
    @property
    def dtype(self) -> np.dtype:
        """The data type of the underlying array."""
        return self.protected_array().dtype

    @override
    @property
    def device(self) -> str:
        """The device of the underlying array."""
        return self.protected_array().device

    @override
    @property
    def ndim(self) -> int:
        """Number of batch dimensions (excluding category axis)."""
        return self.protected_array().ndim - self.protected_axes

    @override
    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape (excluding category axis)."""
        return self.protected_array().shape[: -self.protected_axes]

    @property
    def protected_shape(self) -> tuple[int, ...]:
        """Shape of the protected trailing axes."""
        return self.protected_array().shape[-self.protected_axes :]

    @override
    @property
    def size(self) -> int:
        """The total number of distributions."""
        return int(np.prod(self.shape)) if self.shape else 1

    @override
    @property
    def flags(self) -> ArrayFlagsLike:
        return self.protected_array().flags

    def _validate_result_preserves_protected_axes(self, result: np.ndarray) -> None:
        if result.ndim < self.protected_axes:
            msg = "Operation removed protected trailing axes."
            raise ValueError(msg)
        if result.shape[-self.protected_axes :] != self.protected_shape:
            msg = "Operation modified protected trailing axes."
            raise ValueError(msg)

    def _index_with_protected_axes(self, index: ToIndices) -> tuple[Any, ...]:
        index_tuple = index if isinstance(index, tuple) else (index,)
        return (*index_tuple, *(slice(None),) * self.protected_axes)

    def _coerce_assignment_value(self, value: object) -> np.ndarray:
        value_array = value.protected_array() if isinstance(value, type(self)) else np.asarray(value)

        if value_array.ndim < self.protected_axes:
            msg = "Assigned value must include all protected trailing axes."
            raise ValueError(msg)

        if value_array.shape[-self.protected_axes :] != self.protected_shape:
            msg = "Assigned value modifies protected trailing axes."
            raise ValueError(msg)

        return value_array

    def __getitem__(self, index: ToIndices, /) -> Self:
        """Return a subset while preserving protected trailing axes."""
        full_index = self._index_with_protected_axes(index)
        result = self.protected_array()[full_index]

        if not isinstance(result, np.ndarray):
            msg = "Indexing cannot remove protected trailing axes."
            raise IndexError(msg)

        self._validate_result_preserves_protected_axes(result)
        return self.with_protected_array(result)

    def __setitem__(self, index: ToIndices, value: object, /) -> None:
        """Set a subset while preserving protected trailing axes."""
        full_index = self._index_with_protected_axes(index)

        target = self.protected_array()[full_index]
        if not isinstance(target, np.ndarray):
            msg = "Indexing cannot remove protected trailing axes."
            raise IndexError(msg)

        self._validate_result_preserves_protected_axes(target)
        self.protected_array()[full_index] = self._coerce_assignment_value(value)

    def _postprocess_ufunc_result(self, result: np.ndarray, *, ufunc: np.ufunc, method: str) -> np.ndarray:
        del ufunc, method
        return result

    def _wrap_ufunc_result(self, result: np.ndarray, *, ufunc: np.ufunc, method: str) -> Self | np.ndarray:
        if result.ndim < self.protected_axes:
            return result
        if result.shape[-self.protected_axes :] != self.protected_shape:
            return result
        processed = self._postprocess_ufunc_result(result, ufunc=ufunc, method=method)
        return self.with_protected_array(processed)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> object:
        """Handle ufuncs while preserving protected trailing axes."""
        if method != "__call__":
            return NotImplemented

        out = kwargs.get("out")
        if out is not None:
            outs = out if isinstance(out, tuple) else (out,)
            kwargs["out"] = tuple(o.protected_array() if isinstance(o, type(self)) else o for o in outs)

        converted_inputs = tuple(x.protected_array() if isinstance(x, type(self)) else x for x in inputs)
        result = getattr(ufunc, method)(*converted_inputs, **kwargs)

        if out is not None:
            return out

        if isinstance(result, tuple):
            return tuple(
                self._wrap_ufunc_result(r, ufunc=ufunc, method=method) if isinstance(r, np.ndarray) else r
                for r in result
            )

        if isinstance(result, np.ndarray):
            return self._wrap_ufunc_result(result, ufunc=ufunc, method=method)

        return result

    def __array_function__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Handle selected NumPy functions with protected-axis semantics."""
        return array_function(func, types, args, kwargs)
