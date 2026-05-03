"""Utilities for representations with protected trailing JAX-array axes."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload, override

import jax
import jax.numpy as jnp

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    protected_shape,
    value_ndim,
    value_shape,
)
from probly.representation.jax_like import JaxLikeImplementation

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import numpy as np
    from numpy.typing import DTypeLike

    from probly.representation.array_like import ToIndices


type JaxProtectedValue = JaxLikeImplementation | jax.Array


def _validate_field_ndim(name: str, ndim: int, protected_axes: int) -> None:
    if ndim < protected_axes:
        msg = f"Protected field {name!r} has ndim {ndim}, expected >= {protected_axes}."
        raise ValueError(msg)


class JaxAxisProtected[T: JaxLikeImplementation | jax.Array](JaxLikeImplementation[T]):
    """ABC for representations with one or multiple protected JAX-array fields."""

    protected_axes: ClassVar[dict[str, int]] = {}
    # Reserved: not currently consulted because JAX has no __jax_function__-equivalent
    # dispatch hook. Kept for parity with torch/array; will activate if such a hook
    # is added in the future.
    permitted_functions: ClassVar[set[Callable[..., Any]]] = set()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        if cls is JaxAxisProtected:
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

    def _postprocess_protected_values[V: JaxProtectedValue](self, values: dict[str, V], func: Callable) -> dict[str, V]:
        """Optionally postprocess protected values based on the triggering function.

        Reserved: not currently consulted because JAX has no
        ``__jax_function__``-equivalent dispatch hook. Kept for parity with
        torch/array; will activate if such a hook is added in the future.
        """
        del func
        return values

    def _rotate_sample_axis(self, source_axis: int, dest_axis: int) -> Self:
        """Return a copy with batch axis permuted, preserving protected trailing axes.

        Subclasses must override this to permute their underlying protected
        value(s) and rebuild the wrapper. This mirrors what
        ``__torch_function__`` / ``__array_function__`` do implicitly on
        torch/array: it lets generic ``JaxArraySample.samples`` rotate the
        sample axis without ``jnp.moveaxis`` ever seeing the wrapper.

        Args:
            source_axis: Current batch axis to rotate from.
            dest_axis: Target batch axis to rotate to.

        Returns:
            A new instance of the same type with the batch axis rotated.
        """
        msg = f"{type(self).__name__} must override _rotate_sample_axis."
        raise NotImplementedError(msg)

    @overload
    def protected_values(self) -> dict[str, JaxProtectedValue]: ...

    @overload
    def protected_values(self, func: Callable) -> dict[str, JaxProtectedValue] | None: ...

    def protected_values(self, func: Callable | None = None) -> dict[str, JaxProtectedValue] | None:
        """Return all protected field values as-is.

        Optionally takes the function that triggered the call for context.
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

    def with_protected_values(self, values: dict[str, JaxProtectedValue]) -> Self:
        """Return a copy with updated protected field values."""
        current_values = self.protected_values()
        updates: dict[str, object] = {}

        for name in type(self).protected_axes:
            updates[name] = values.get(name, current_values[name])

        return cast("Self", replace(cast("Any", self), **updates))

    def with_protected_value(self, value: JaxProtectedValue) -> Self:
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

    @property
    def dtype(self) -> Any:  # noqa: ANN401
        """Return the dtype of the primary protected value."""
        return cast("Any", self.protected_value()).dtype

    @property
    def device(self) -> Any:  # noqa: ANN401
        """Return the device of the primary protected value."""
        return cast("Any", self.protected_value()).device

    @property
    def ndim(self) -> int:
        """Return the number of batch dimensions."""
        primary_name = type(self).primary_protected_name()
        axes = type(self).protected_axes[primary_name]
        return len(batch_shape(value_shape(self.protected_value()), axes))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the batch shape of the primary protected value."""
        primary_name = type(self).primary_protected_name()
        axes = type(self).protected_axes[primary_name]
        return batch_shape(value_shape(self.protected_value()), axes)

    @property
    def protected_shape(self) -> tuple[int, ...]:
        """Protected trailing shape of the primary field."""
        primary_name = type(self).primary_protected_name()
        axes = type(self).protected_axes[primary_name]
        return protected_shape(value_shape(self.protected_value()), axes)

    @property
    def size(self) -> int:
        """Return the total number of batch elements."""
        out = 1
        for dim in self.shape:
            out *= dim
        return out

    @property
    def T(self) -> Self:  # noqa: N802
        """Transposed view across batch dimensions.

        For ``ndim < 2`` this is a no-op and returns ``self``. This intentionally
        differs from :attr:`mT` (which raises) because ``T`` is defined for any
        rank in the array API, whereas matrix transpose requires at least two
        dimensions. The asymmetry mirrors the torch-side ``TorchAxisProtected``.
        """
        if self.ndim < 2:
            return self
        values = self.protected_values()
        transposed: dict[str, JaxProtectedValue] = {}
        for name, axes in type(self).protected_axes.items():
            value = values[name]
            batch_ndim = value_ndim(value) - axes
            permutation = (*range(batch_ndim - 1, -1, -1), *range(batch_ndim, value_ndim(value)))
            transposed[name] = cast("JaxProtectedValue", jnp.transpose(cast("Any", value), permutation))
        return self.with_protected_values(transposed)

    @property
    def mT(self) -> Self:  # noqa: N802
        """Matrix-transposed view (swap last two batch axes)."""
        if self.ndim < 2:
            msg = "mT requires at least 2 batch dimensions."
            raise ValueError(msg)
        values = self.protected_values()
        transposed: dict[str, JaxProtectedValue] = {}
        for name, axes in type(self).protected_axes.items():
            value = values[name]
            batch_ndim = value_ndim(value) - axes
            transposed[name] = cast(
                "JaxProtectedValue",
                jnp.swapaxes(cast("Any", value), batch_ndim - 2, batch_ndim - 1),
            )
        return self.with_protected_values(transposed)

    def _index_with_protected_axes(self, index: ToIndices, protected_axes_count: int) -> tuple[Any, ...]:
        index_tuple = index if isinstance(index, tuple) else (index,)
        return (*index_tuple, *(slice(None),) * protected_axes_count)

    @override
    def __getitem__(self, index: ToIndices, /) -> Self:
        values = self.protected_values()
        indexed: dict[str, JaxProtectedValue] = {}

        for name, axes in type(self).protected_axes.items():
            full_index = self._index_with_protected_axes(index, axes)
            result = cast("Any", values[name])[full_index]

            if axes == 0 and not hasattr(result, "ndim"):
                result = jnp.asarray(result)

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
        msg = f"{type(self).__name__} is immutable; use functional updates."
        raise TypeError(msg)

    def to_device(self, device: Any, /, *, stream: int | Any | None = None) -> Self:  # noqa: ANN401
        """Move the underlying values to the specified device."""
        if stream is not None:
            msg = "stream argument of to_device()"
            raise NotImplementedError(msg)

        values = self.protected_values()
        updates: dict[str, JaxProtectedValue] = {}
        for name, value in values.items():
            updates[name] = cast("JaxProtectedValue", cast("Any", value).to_device(device))
        return self.with_protected_values(updates)

    def numpy(self, *, force: bool = False) -> np.ndarray:
        """Convert the primary protected value to a numpy array.

        Args:
            force: Ignored on JAX (kept for API parity with the torch backend).

        Returns:
            The primary protected value as a NumPy array.
        """
        del force
        if len(type(self).protected_axes) != 1:
            msg = "Cannot convert multi-field protected object to a single numpy array."
            raise TypeError(msg)
        import numpy as np  # noqa: PLC0415

        return np.asarray(self.protected_value())

    @override
    def __array__(self, dtype: DTypeLike | None = None, /, *, copy: bool | None = None) -> np.ndarray:
        array = self.numpy()
        if dtype is not None:
            return array.astype(dtype, copy=bool(copy))
        if copy:
            return array.copy()
        return array
