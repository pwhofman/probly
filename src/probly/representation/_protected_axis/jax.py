"""Utilities for representations with protected trailing tensor axes."""

from __future__ import annotations

from abc import ABC
from dataclasses import replace
from inspect import isabstract
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload, override

import numpy as np
import jax
from jax import numpy as jnp

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    protected_shape,
    value_ndim,
    value_shape,
)
from probly.representation._protected_axis.jax_functions import jax_function
from probly.representation.jax_like import JaxLike, JaxLikeImplementation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

    from numpy.typing import DTypeLike, NDArray

    from probly.representation.array_like import ToIndices


type JaxProtectedValue = JaxLike[Any] | jax.Array | np.ndarray


def _validate_field_ndim(name: str, ndim: int, protected_axes: int) -> None:
    if ndim < protected_axes:
        msg = f"Protected field {name!r} has ndim {ndim}, expected >= {protected_axes}."
        raise ValueError(msg)

class JaxAxisProtected[J: JaxLike | jax.Array | np.ndarray](JaxLikeImplementation[J], ABC):
    """ABC for representation with protected jax-like felds and optional NumPy sidecars."""

    protected_axes: ClassVar[dict[str, int]] = {}
    permitted_functions: ClassVar[set[Callable[..., Any]]] = set()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if cls is JaxAxisProtected or isabstract(cls):
            return

        axes = getattr(cls, "protected_axes", None)
        if not isinstance(axes, dict) or len(axes) == 0:
            msg = f"{cls.__name__} must define protected_axes as a non-empty dict[str, int]."
            return TypeError(msg)

        for name, value in axes.items():
            if not isinstance(name, str) or len(name) == 0:
                msg = f"{cls.__name__}.protected_axes must use non-empty string keys."
                raise TypeError(msg)
            if not isinstance(value, int) or value < 0:
                msg = f"{cls.__name__}.protected_axes[{name}] must be a non-negative integer."
                raise TypeError(msg)
            if not hasattr(cls, "__annotations__") or name not in cls.__annotations__:
                msg = f"{cls.__name__}.protected_axes refers to unkown field {name!r}."
                raise TypeError(msg)

        permitted_functions = getattr(cls, "permitted_functions", set())
        if not isinstance(permitted_functions, set) or not all(callable(func) for func in permitted_functions):
            msg = f"{cls.__name}.permitted_functions must be a set of callables."
            raise TypeError(msg)
        cls.permitted_functions = set(permitted_functions)

        @classmethod
        def primary_protected_name(cls) -> str:
            return next(iter(cls.protected_axes))

        def _postprocess_protected_values[V: JaxProtectedValue](
            self, values: dict[str,V], func: Callable
        ) -> dict[str,V]:
            """Optionally postprocess protected values based on the triggering function."""
            del func
            return values

        @overload
        def protected_values(self) -> dict[str, JaxProtectedValue]: ...

        @overload
        def procted_values(self, func: Callable) -> dict[str, JaxProtectedValue] | None: ...

        def protected_values(self, func: Callable | None = None) -> dict[str, JaxProtectedValue] | None:
            """Return all protected field values as-is.

            Optionally takes the jax function that triggered the call for context.
            This can be used to confitionally modify the returned values or prevent them from being accessed.
            """
            if func is not None and func not in type(self).permitted_functions:
                return None

            values: dict[str, JaxProtectedValue] = {}
            primary_name = type(self).primary_protected_name()
            primary_batch: tuple[int, ...] | None = None

            for name, axes in type(self).protected_axes.items():
                value = cast("JaxProtectedValue", getattr(self, name))
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

            if func is not None:
                values = self._posprocess_protected_values(values, func)

            return values

        def protected_value(self) -> JaxProtectedValue:
            """ Return the primary protected value."""
            primary_name = type(self).primary_protected_name()
            return self.protected_values()[primary_name]

        def _jax_protected_value(self) -> JaxLike[Any] | jax.Array:
            """Return the first jax-like protected value."""
            for value in self.protected_values().values():
                if not isinstance(value, np.ndarray):
                    return value

            msg = "No torch-like protected values available."
            raise TypeError(msg)

        def with_protected_values(
            self,
            values: dict[str, JaxProtectedValue],
            func: Callable | None = None,   # noqa: ARG002
        ) -> JaxAxisProtected[J]:
            """Return a copy with updated protected field values."""
            current_values = self.protected_values()
            updates: dict[str, object] = {}

            for name in type(self).protected_axes:
                updates[name] = values.get(name, current_values[name])

            return replace(self, **updates) # ty:ignore[invalid-argument-type]

        @override
        def __len__(self) -> int:
            if self.ndim == 0:
                msg = "len() of unsized distribution"
                raise TypeError(msg)
            return len(cast("Any", self.protected_values()))

        @override
        def __iter__(self) -> Iterator[Any]:
            for index in range(len(self)):
                yield self[index]

        @override
        def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
            return cast("Any", self._jax_protected_value()).__array_namespace__(api_version=api_version)

        @override
        @property
        def dtypes(self) -> jax.dtypes:
            return cast("jax.dtypes", self._jax_protected_value().dtypes)

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

        @override
        @property
        def protected_shape(self) -> tuple[int, ...]:
            """Protected trailing shape of the primary field."""
            primary_name = type(self).primary_protected_name()
            axes = type(self).protected_axes[primary_name]
            return protected_shape(value_shape(self.protected_value()), axes)

        @overload
        def size(self, dim: int) -> int: ...

        @overload
        def size(self, dim: None = ...) -> jax.Array.size: ...

        @override
        def size(self, dim: int | None = None) -> jax.Array.size:
            if dim is None:
                return jnp.array.size(self.shape)

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
            return self.transpose(-2, -1)

        @override
        @property
        def mH(self) -> Self:
            if self.ndim < 2:
                msg = "mH requires at least 2 batch dimensions."
                raise ValueError(msg)
            return cast("Self", jax.adjoint(cast("Any", self)))

        def _index_with_protected_axes(self, index: ToIndices, protected_axes_count: int) -> tuple[Any, ...]:
            index_tuple = index if isinstance(index, tuple) else (index,)
            return (*index_tuple, *(slice(None),) * protected_axes_count)

        def _coerce_assignment_value(self, value: object) -> dict[str, object]:
            field_names = tuple(type(self).protected_axes.keys())

            if isinstance(value, type(self)):
                candidate_values: dict[str, object] = value.protected_values() # ty:ignore[invalid-assignment]
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
        def __getitem__(self, index: ToIndices, /) -> JaxAxisProtected[J]:
            values = self.protected_values()
            indexed: dict[str, JaxProtectedValue] = {}

            for name, axes in type(self).protected_axes.items():
                full_index = self._index_with_protected_axes(index, axes)
                value = values[name]
                result = cast("Any", value)[full_index]

                if axes == 0 and not hasattr(result, "ndim"):
                    result = (
                        np.asarray(result, dtype=value.dtypes) if isinstance(value, np.ndarray) else jnp.array(result)
                    )

                result_ndim = value_ndim(result)
                result_shape = value_shape(result)
                _validate_field_ndim(name, result_ndim, axes)

                original_shape = value_shape(values[name])
                if protected_shape(result_shape, axes) != protected_shape(original_shape, axes):
                    msg = f"Indexing field {name!r} modified protected trailing axes."
                    raise IndexError(msg)

                indexed[name] = cast("JaxProtectedValue", result)

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
            updates: dict[str, JaxProtectedValue] = {}
            changed = False

            for name, value in values.items():
                if isinstance(value, np.ndarray):
                    updates[name] = value
                    continue

                converted = cast("Any", value).to(**args, **kwargs)
                updates[name] = cast("JaxProtectedValue", converted)
                changed = changed or converted is not value

            if not changed:
                return self

            return self.with_protected_values(updates) # ty:ignore[invalid-return-type]

        def __jax_like__(
            self,
            dtypes: jax.dtypes | None = None,
            /,
            *,
            copy: bool = False,
        ) -> JaxLikeImplementation[Any]:
            return self.to(dtypes=dtypes, copy=copy)

        @classmethod
        def __jax_function(
            cls,
            func: Callable,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any: # noqa: ANN401
            del cls
            return jax_function(func, types, args, {} if kwargs is None else kwargs)

        @override
        def numpy(self, *, force: bool = False) -> np.ndarray:
            if len(type(self).protected_axes) != 1:
                msg = "Cannot convert multi-field protected object to a single numpy array."
                raise TypeError(msg)
            value = self.protected_value()
            if isinstance(value, np.ndarray):
                return value.copy() if force else value
            return cast("Any", value).numpy(force=force)

        def __array__(self, dtype: DTypeLike | None = None, /, *, copy:bool | None = None) -> NDArray[Any]:
            array = self.numpy(force=dtype is not None or bool(copy))
            if dtype is not None:
                return array.astype(dtype, copy=bool(copy))
            if copy:
                return array.copy()
            return array

        def reshape(self, *shape: int | tuple[int, ...]) -> Self:
            """Retur a copy with reshaped protected values."""
            if len(shape) == 1 and isinstance(shape[0], tuple):
                shape = shape[0]

            return jax.reshape(self, shape) # ty:ignore[invalid-return-type, invalid-argument-type]

        def gather(self, dim: int, index: jax.Array) -> Self:
            """Return a copy with gathered protected values along a batch dimension."""
            return jax.gather(self, dim, index) # ty:ignore[no-matching-overload]
