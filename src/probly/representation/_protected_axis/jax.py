"""Utilities for representations with protected trailing jax array axes."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, replace
from inspect import isabstract
import math
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload, override

import jax
from jax import numpy as jnp
import numpy as np

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    protected_shape,
    value_ndim,
    value_shape,
)
from probly.representation._protected_axis.jax_functions import jax_function
from probly.representation.jax_functions import (
    jax_conj,
    jax_reshape,
    jax_swapaxes,
    jax_take_along_axis,
)
from probly.representation.jax_like import JaxLike, JaxLikeImplementation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import ModuleType

    from jax.sharding import Sharding
    from numpy.typing import DTypeLike, NDArray

    from probly.representation.array_like import ToIndices


type JaxProtectedValue = JaxLike[Any] | jax.Array | np.ndarray


def _validate_field_ndim(name: str, ndim: int, protected_axes: int) -> None:
    if ndim < protected_axes:
        msg = f"Protected field {name!r} has ndim {ndim}, expected >= {protected_axes}."
        raise ValueError(msg)


class JaxAxisProtected[J: JaxLike | jax.Array | np.ndarray](JaxLikeImplementation[J], ABC):
    """ABC for representations with protected jax-like fields and optional NumPy sidecars."""

    protected_axes: ClassVar[dict[str, int]] = {}
    permitted_functions: ClassVar[set[Callable[..., Any]]] = set()
    allow_types: ClassVar[tuple[type, ...]] = (jnp.ndarray, jnp.generic, float, int)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if cls is JaxAxisProtected or isabstract(cls):
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
                msg = f"{cls.__name__}.protected_axes[{name}] must be a non-negative integer."
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

    def _postprocess_protected_values[V: JaxProtectedValue](self, values: dict[str, V], func: Callable) -> dict[str, V]:
        """Optionally postprocess protected values based on the triggering function."""
        del func
        return values

    @overload
    def protected_values(self) -> dict[str, JaxProtectedValue]: ...

    @overload
    def protected_values(self, func: Callable) -> dict[str, JaxProtectedValue] | None: ...

    def protected_values(self, func: Callable | None = None) -> dict[str, JaxProtectedValue] | None:
        """Return all protected field values as-is.

        Optionally takes the jax function that triggered the call for context.
        This can be used to conditionally modify the returned values or prevent them from being accessed.
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
            values = self._postprocess_protected_values(values, func)

        return values

    def protected_value(self) -> JaxProtectedValue:
        """Return the primary protected value."""
        primary_name = type(self).primary_protected_name()
        return self.protected_values()[primary_name]

    def _jax_protected_value(self) -> JaxLike[Any] | jax.Array:
        """Return the first jax-like protected value."""
        for value in self.protected_values().values():
            if not isinstance(value, np.ndarray):
                return value

        msg = "No jax-like protected values available."
        raise TypeError(msg)

    def with_protected_values(
        self,
        values: dict[str, JaxProtectedValue],
        func: Callable | None = None,  # noqa: ARG002
    ) -> JaxAxisProtected[J]:
        """Return a copy with updated protected field values."""
        current_values = self.protected_values()
        updates: dict[str, object] = {}

        for name in type(self).protected_axes:
            updates[name] = values.get(name, current_values[name])

        return replace(self, **updates)  # ty:ignore[invalid-argument-type]

    @override
    def __len__(self) -> int:
        if self.ndim == 0:
            msg = "len() of unsized distribution"
            raise TypeError(msg)
        return len(cast("Any", self.protected_value()))

    @override
    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    @override
    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        return cast("Any", self._jax_protected_value()).__array_namespace__(api_version=api_version)

    @override
    @property
    def dtype(self) -> jnp.dtype:
        return cast("jnp.dtype", self._jax_protected_value().dtype)

    @override
    @property
    def device(self) -> jax.Device:
        return cast("jax.Device", self._jax_protected_value().device)

    @property
    def sharding(self) -> Any:  # noqa: ANN401
        """The sharding of the underlying array."""
        return self._jax_protected_value().sharding

    def devices(self) -> set[jax.Device]:
        """Return the set of devices the array lives on."""
        return self._jax_protected_value().devices()

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
    def size(self, dim: None = ...) -> int: ...

    @override
    def size(self, dim: int | None = None) -> int:
        if dim is None:
            return math.prod(self.shape)

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
        return cast("Self", jax_swapaxes(cast("Any", self), -2, -1))

    @override
    @property
    def mH(self) -> Self:
        if self.ndim < 2:
            msg = "mH requires at least 2 batch dimensions."
            raise ValueError(msg)
        return cast("Self", jax_conj(cast("Any", self.mT)))

    def _index_with_protected_axes(self, index: ToIndices, protected_axes_count: int) -> tuple[Any, ...]:
        index_tuple = index if isinstance(index, tuple) else (index,)
        return (*index_tuple, *(slice(None),) * protected_axes_count)

    def _coerce_assignment_value(self, value: object, *, validate_shape: bool = True) -> dict[str, object]:
        field_names = tuple(type(self).protected_axes.keys())

        if isinstance(value, type(self)):
            candidate_values: dict[str, object] = value.protected_values()  # ty:ignore[invalid-assignment]
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

        if not validate_shape:
            return candidate_values

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

    def _gather_protected_values(self, index: ToIndices) -> JaxAxisProtected[J]:
        """Return a copy with the indexed batch slice of every protected field."""
        values = self.protected_values()
        indexed: dict[str, JaxProtectedValue] = {}

        for name, axes in type(self).protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            value = values[name]
            result = cast("Any", value)[full_index]

            if axes == 0 and not hasattr(result, "ndim"):
                result = np.asarray(result, dtype=value.dtype) if isinstance(value, np.ndarray) else jnp.array(result)

            result_ndim = value_ndim(result)
            result_shape = value_shape(result)
            _validate_field_ndim(name, result_ndim, axes)

            original_shape = value_shape(values[name])
            if protected_shape(result_shape, axes) != protected_shape(original_shape, axes):
                msg = f"Indexing field {name!r} modified protected trailing axes."
                raise IndexError(msg)

            indexed[name] = cast("JaxProtectedValue", result)

        return self.with_protected_values(indexed)

    def _scatter_protected_values(
        self,
        index: ToIndices,
        method: str,
        value: object,
        kwargs: dict[str, Any],
    ) -> JaxAxisProtected[J]:
        """Return a copy with an out-of-place ``jax.Array.at`` update applied to every protected field."""
        values = self.protected_values()
        candidate_values = self._coerce_assignment_value(value, validate_shape=method == "set")
        updates: dict[str, JaxProtectedValue] = {}

        for name, axes in type(self).protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            field = values[name]

            if isinstance(field, np.ndarray):
                if method != "set":
                    msg = f"Indexed update {method!r} is not supported for NumPy field {name!r}."
                    raise TypeError(msg)
                candidate = candidate_values[name]
                if field.dtype == object and isinstance(candidate, np.ndarray) and candidate.ndim == 0:
                    candidate = cast("Any", candidate).item()
                copied = field.copy()
                copied[full_index] = candidate
                updates[name] = copied
                continue

            updated = getattr(cast("Any", field).at[full_index], method)(candidate_values[name], **kwargs)
            updates[name] = cast("JaxProtectedValue", updated)

        return self.with_protected_values(updates)

    @property
    def at(self) -> JaxAxisProtectedIndexUpdateHelper[J]:
        """Return the out-of-place indexed update helper, mirroring ``jax.Array.at``."""
        return JaxAxisProtectedIndexUpdateHelper(self)

    @override
    def __getitem__(self, index: ToIndices, /) -> JaxAxisProtected[J]:
        return self._gather_protected_values(index)

    @override
    def __setitem__(self, index: ToIndices, value: object, /) -> None:
        del index, value
        msg = "JAX arrays are immutable, use x = x.at[index].set(value) instead."
        raise TypeError(msg)

    @override
    def astype(
        self,
        dtype: DTypeLike | None,
        copy: bool = False,
        device: jax.Device | Sharding | None = None,
    ) -> Self:
        values = self.protected_values()
        updates: dict[str, JaxProtectedValue] = {}
        changed = False

        for name, value in values.items():
            if isinstance(value, np.ndarray):
                updates[name] = value
                continue

            converted = cast("Any", value).astype(dtype, copy=copy, device=device)
            updates[name] = cast("JaxProtectedValue", converted)
            changed = changed or converted is not value

        if not changed:
            return self

        return self.with_protected_values(updates)  # ty:ignore[invalid-return-type]

    def __jax_like__(
        self,
        dtype: jnp.dtype | None = None,
        /,
        *,
        device: jax.Device | Sharding | None = None,
        copy: bool = False,
    ) -> JaxLikeImplementation[Any]:
        return self.astype(dtype=dtype, copy=copy, device=device)

    @classmethod
    def __jax_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        del cls
        return jax_function(func, types, args, {} if kwargs is None else kwargs)

    def numpy(self, *, force: bool = False) -> np.ndarray:
        if len(type(self).protected_axes) != 1:
            msg = "Cannot convert multi-field protected object to a single numpy array."
            raise TypeError(msg)
        value = self.protected_value()
        if isinstance(value, np.ndarray):
            return value.copy() if force else value
        return np.array(value) if force else np.asarray(value)

    def __array__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> NDArray[Any]:
        array = self.numpy(force=dtype is not None or bool(copy))
        if dtype is not None:
            return array.astype(dtype, copy=bool(copy))
        if copy:
            return array.copy()
        return array

    def _postprocess_arithmetic_result(self, values: dict[str, JaxProtectedValue]) -> dict[str, JaxProtectedValue]:
        """Optionally postprocess field values produced by ``+``/``-``.

        Called with whichever fields ``_combine`` just computed (all of them for two same-type
        operands, only the primary field for a scalar operand). The default is a no-op; override to
        clamp results back into a valid domain, e.g. keeping Dirichlet concentration parameters positive.
        """
        return values

    def _combine(self, other: object, *, subtract: bool, reflected: bool) -> Self:
        def op(left: Any, right: Any) -> Any:  # noqa: ANN401
            return left - right if subtract else left + right

        if isinstance(other, type(self)):
            values = self.protected_values()
            other_values = other.protected_values()
            combined = {
                name: op(other_values[name], cast("Any", value))
                if reflected
                else op(cast("Any", value), other_values[name])
                for name, value in values.items()
            }
        elif isinstance(other, self.allow_types):
            primary_name = type(self).primary_protected_name()
            primary_value = cast("Any", self.protected_values()[primary_name])
            combined = {primary_name: op(other, primary_value) if reflected else op(primary_value, other)}
        else:
            return NotImplemented

        combined = self._postprocess_arithmetic_result(combined)
        return cast("Self", self.with_protected_values(combined))

    def __add__(self, other: object) -> Self:
        """Add another instance of the same type (fields combine pairwise) or a scalar/array.

        JAX has no numpy-style operator override protocol, so this (and the other arithmetic dunders
        here) is what makes ``+``/``-`` work at all for protected-axis subclasses.
        """
        return self._combine(other, subtract=False, reflected=False)

    def __radd__(self, other: object) -> Self:
        """Support ``scalar + instance``; addition here is symmetric."""
        return self._combine(other, subtract=False, reflected=False)

    def __sub__(self, other: object) -> Self:
        """Subtract another instance of the same type or a scalar/array."""
        return self._combine(other, subtract=True, reflected=False)

    def __rsub__(self, other: object) -> Self:
        """Support ``scalar - instance``."""
        return self._combine(other, subtract=True, reflected=True)

    @override
    def reshape(self, *args: int | Sequence[int], order: str = "C") -> Self:
        """Return a copy with reshaped protected values."""
        shape = cast("Sequence[int]", args[0] if len(args) == 1 and not isinstance(args[0], int) else args)

        return cast("Self", jax_reshape(cast("Any", self), tuple(shape), order=order))

    def take_along_axis(self, indices: jax.Array, axis: int = -1) -> Self:
        """Return a copy with gathered protected values along a batch dimension."""
        return cast("Self", jax_take_along_axis(cast("Any", self), indices, axis))


@dataclass(frozen=True, slots=True)
class JaxAxisProtectedIndexUpdateRef[J: JaxLike | jax.Array | np.ndarray]:
    """Update helper bound to one batch index, mirroring ``jax.Array.at[index]``."""

    owner: JaxAxisProtected[J]
    index: ToIndices

    def get(self) -> JaxAxisProtected[J]:
        """Return a copy holding the indexed slice of every protected field, same as ``owner[index]``."""
        return self.owner[self.index]

    def set(self, value: object, **kwargs: Any) -> JaxAxisProtected[J]:  # noqa: ANN401
        """Return a copy with the indexed batch entries replaced by ``value``."""
        return self.owner._scatter_protected_values(self.index, "set", value, kwargs)  # noqa: SLF001

    def add(self, value: object, **kwargs: Any) -> JaxAxisProtected[J]:  # noqa: ANN401
        """Return a copy with ``value`` added to the indexed batch entries."""
        return self.owner._scatter_protected_values(self.index, "add", value, kwargs)  # noqa: SLF001

    def subtract(self, value: object, **kwargs: Any) -> JaxAxisProtected[J]:  # noqa: ANN401
        """Return a copy with ``value`` subtracted from the indexed batch entries."""
        return self.owner._scatter_protected_values(self.index, "subtract", value, kwargs)  # noqa: SLF001

    def multiply(self, value: object, **kwargs: Any) -> JaxAxisProtected[J]:  # noqa: ANN401
        """Return a copy with the indexed batch entries multiplied by ``value``."""
        return self.owner._scatter_protected_values(self.index, "multiply", value, kwargs)  # noqa: SLF001

    def divide(self, value: object, **kwargs: Any) -> JaxAxisProtected[J]:  # noqa: ANN401
        """Return a copy with the indexed batch entries divided by ``value``."""
        return self.owner._scatter_protected_values(self.index, "divide", value, kwargs)  # noqa: SLF001

    def power(self, value: object, **kwargs: Any) -> JaxAxisProtected[J]:  # noqa: ANN401
        """Return a copy with the indexed batch entries raised to the power of ``value``."""
        return self.owner._scatter_protected_values(self.index, "power", value, kwargs)  # noqa: SLF001

    def min(self, value: object, **kwargs: Any) -> JaxAxisProtected[J]:  # noqa: ANN401
        """Return a copy with the elementwise minimum of the indexed batch entries and ``value``."""
        return self.owner._scatter_protected_values(self.index, "min", value, kwargs)  # noqa: SLF001

    def max(self, value: object, **kwargs: Any) -> JaxAxisProtected[J]:  # noqa: ANN401
        """Return a copy with the elementwise maximum of the indexed batch entries and ``value``."""
        return self.owner._scatter_protected_values(self.index, "max", value, kwargs)  # noqa: SLF001


@dataclass(frozen=True, slots=True)
class JaxAxisProtectedIndexUpdateHelper[J: JaxLike | jax.Array | np.ndarray]:
    """Helper returned by :attr:`JaxAxisProtected.at`, mirroring ``jax.Array.at``."""

    owner: JaxAxisProtected[J]

    def __getitem__(self, index: ToIndices, /) -> JaxAxisProtectedIndexUpdateRef[J]:
        """Bind a batch index, which never addresses the protected trailing axes."""
        return JaxAxisProtectedIndexUpdateRef(self.owner, index)
