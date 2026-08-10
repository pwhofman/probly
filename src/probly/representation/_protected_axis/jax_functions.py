"""Jax-function implementations for protected-axis values."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from inspect import BoundArguments, signature
from typing import TYPE_CHECKING, Any, Protocol, cast, overload, runtime_checkable

import jax
from jax import numpy as jnp
import numpy as np

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    normalize_axes,
    normalize_axis,
    protected_shape,
    value_ndim,
    value_shape,
)
from probly.representation.jax_functions import (
    jax_astype,
    jax_average,
    jax_concatenate,
    jax_conj,
    jax_copy,
    jax_expand_dims,
    jax_matrix_transpose,
    jax_mean,
    jax_moveaxis,
    jax_reshape,
    jax_squeeze,
    jax_stack,
    jax_std,
    jax_sum,
    jax_swapaxes,
    jax_take_along_axis,
    jax_transpose,
    jax_var,
)
from probly.representation.jax_like import JaxLike
from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable


type JaxProtectedValue = JaxLike[Any] | jax.Array | np.ndarray


class JaxAxisProtectedCreator(Protocol):
    """Protocol for rebuilding protected-axis representations."""

    def __call__(self, values: dict[str, JaxProtectedValue]) -> Any:  # noqa: ANN401
        """Create object from updated protected values."""


@runtime_checkable
class _SupportsProtectedInternals(Protocol):
    protected_axes: dict[str, int]
    permitted_functions: set[Callable[..., Any]]

    @overload
    def protected_values(self) -> dict[str, JaxProtectedValue]: ...

    @overload
    def protected_values(self, func: Callable) -> dict[str, JaxProtectedValue] | None: ...

    def protected_values(self, func: Callable | None = None) -> dict[str, JaxProtectedValue] | None:
        """Return protected field values."""

    def with_protected_values(self, values: dict[str, JaxProtectedValue], func: Callable | None = None) -> Any:  # noqa: ANN401
        """Create a copy with updated protected values."""


@dataclass(frozen=True, slots=True)
class JaxAxisProtectedInternals:
    """Internal representation for one protected-axis object."""

    create: JaxAxisProtectedCreator
    values: dict[str, JaxProtectedValue]
    protected_axes: dict[str, int]
    primary_name: str
    owner_type: type[Any]

    @property
    def primary_value(self) -> JaxProtectedValue:
        """Return the primary protected value."""
        return self.values[self.primary_name]

    @property
    def batch_ndim(self) -> int:
        axes = self.protected_axes[self.primary_name]
        return value_ndim(self.primary_value) - axes


@dataclass(frozen=True, slots=True)
class ProtectedValueSequenceInternals:
    """Extracted internals for sequence-based operations."""

    has_protected: bool
    template: JaxAxisProtectedInternals | None
    values_by_field: dict[str, list[object]]


def jax_axis_protected_internals(
    obj: object, func: Callable | None = None, *, check_is_permitted: bool = False
) -> JaxAxisProtectedInternals | None:
    """Extract protected-axis internals from object."""
    if not isinstance(obj, _SupportsProtectedInternals):
        return None
    protected_axes = obj.protected_axes
    if not isinstance(protected_axes, dict) or len(protected_axes) == 0:
        return None
    values = obj.protected_values(func if check_is_permitted else None)  # ty:ignore[invalid-argument-type]

    if values is None:
        return None

    for name, axes in protected_axes.items():
        if name not in values:
            return None
        ndim = value_ndim(values[name])
        if ndim < axes:
            return None

    primary_name = next(iter(protected_axes))

    def create(values: dict[str, JaxProtectedValue]) -> Any:  # noqa: ANN401
        return obj.with_protected_values(values, func)

    owner_type = type(obj)
    return JaxAxisProtectedInternals(
        create=create,
        values=dict(values),
        protected_axes=dict(protected_axes),
        primary_name=primary_name,
        owner_type=owner_type,
    )


def _validate_batch_sync(values: dict[str, JaxProtectedValue], protected_axes: dict[str, int]) -> None:
    expected: tuple[int, ...] | None = None
    for name, value in values.items():
        axes = protected_axes[name]
        ndim = value_ndim(value)
        shape = value_shape(value)
        if ndim < axes:
            msg = f"Operation removed protected trailing axes for field {name!r}."
            raise ValueError(msg)
        current = batch_shape(shape, axes)
        if expected is None:
            expected = current
        elif current != expected:
            msg = "Operation produced inconsistent batch-shapes across protected fields."
            raise ValueError(msg)


def _has_numpy_protected_value(internals: JaxAxisProtectedInternals) -> bool:
    return any(isinstance(value, np.ndarray) for value in internals.values.values())


def _apply_structural_op(
    value: JaxProtectedValue,
    jax_op: Callable[[JaxProtectedValue], object],
    numpy_op: Callable[[np.ndarray], object],
) -> JaxProtectedValue:
    if isinstance(value, np.ndarray):
        return cast("JaxProtectedValue", numpy_op(value))
    return cast("JaxProtectedValue", jax_op(value))


def _apply_unary(
    internals: JaxAxisProtectedInternals,
    op: Callable[[str, JaxProtectedValue, int], JaxProtectedValue],
) -> Any:  # noqa: ANN401
    results: dict[str, JaxProtectedValue] = {}
    for name, value in internals.values.items():
        results[name] = op(name, value, internals.protected_axes[name])

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)


def _extract_protected_value_sequence_internals(
    values: tuple[object, ...], func: Callable | None = None
) -> ProtectedValueSequenceInternals:
    """Extract and align protected values for sequence operations."""
    template: JaxAxisProtectedInternals | None = None
    values_by_field: dict[str, list[object]] = {}
    has_protected = False

    for value in values:
        internals = jax_axis_protected_internals(value, func)
        if internals is None:
            if template is None:
                continue
            for name in template.protected_axes:
                values_by_field[name].append(value)
            continue

        has_protected = True
        if template is None:
            template = internals
            values_by_field = {name: [] for name in internals.protected_axes}
        elif internals.protected_axes != template.protected_axes:
            msg = "All protected inputs must share identical protected_axes definitions."
            raise ValueError(msg)

        for name in template.protected_axes:
            values_by_field[name].append(internals.values[name])

    if not has_protected:
        return ProtectedValueSequenceInternals(False, None, {})

    return ProtectedValueSequenceInternals(True, template, values_by_field)


def _map_batch_axes(
    value: JaxProtectedValue, protected_axes_count: int, batch_axes: tuple[int, ...]
) -> tuple[int, ...]:
    ndim = value_ndim(value)
    batch_ndim = ndim - protected_axes_count
    normalized = normalize_axes(batch_axes, batch_ndim)
    return (*normalized, *range(batch_ndim, ndim))


def _normalize_batch_reduction_dims(axis: object, batch_ndim: int) -> int | tuple[int, ...]:
    if axis is None:
        return tuple(range(batch_ndim))
    if isinstance(axis, int):
        return normalize_axis(axis, batch_ndim)
    if isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
        dim_tuple = cast("tuple[int, ...]", tuple(axis))
        return normalize_axes(dim_tuple, batch_ndim)

    msg = "reduction axis must be None, an int, or a tuple/list of ints."
    raise TypeError(msg)


def _expand_average_weights_for_protected_axes(
    weights: object,
    value: JaxProtectedValue,
    axes_count: int,
) -> object:
    if not isinstance(weights, (jax.Array, np.ndarray)) or axes_count == 0:
        return weights

    batch_ndim = value_ndim(value) - axes_count
    if weights.ndim != batch_ndim:
        return weights

    expanded = cast("JaxProtectedValue", cast("Any", weights).reshape((*weights.shape, *((1,) * axes_count))))
    return _apply_structural_op(
        expanded,
        lambda w: jnp.broadcast_to(cast("Any", w), value_shape(value)),
        lambda w: np.broadcast_to(w, value_shape(value)),
    )


class _JaxFunction(Protocol):
    def __call__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        ...


class _BoundJaxFunction(Protocol):
    def __call__(
        self,
        func: Callable,
        params: BoundArguments,
    ) -> Any:  # noqa: ANN401
        ...


class _BoundJaxFunctionWithInternals(Protocol):
    def __call__(
        self,
        func: Callable,
        params: BoundArguments,
        internals: JaxAxisProtectedInternals,
    ) -> Any:  # noqa: ANN401
        ...


@switchdispatch
def jax_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of jax functions for protected-axis objects."""
    del func, args, kwargs
    return NotImplemented


def jax_function_override(jax_func: _BoundJaxFunction) -> _JaxFunction:
    """Decorator to convert a bound jax function to ``__jax_function__`` shape."""

    @wraps(jax_func)
    def wrapper(
        func: Callable,
        types: tuple[type[Any], ...],  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        params = signature(func).bind(*args, **kwargs)
        params.apply_defaults()
        return jax_func(func, params)

    return wrapper


def _jax_internals_override(
    jax_param_name: str,
    check_is_permitted: bool = False,
) -> Callable[[_BoundJaxFunctionWithInternals], _JaxFunction]:
    """Decorator to convert a function taking a protected-axis argument."""

    def decorator(f: _BoundJaxFunctionWithInternals) -> _JaxFunction:
        @wraps(f)
        def wrapper(func: Callable, params: BoundArguments) -> Any:  # noqa: ANN401
            argument = params.arguments[jax_param_name]
            internals = jax_axis_protected_internals(
                argument,
                func,
                check_is_permitted=check_is_permitted,
            )
            if internals is None:
                return NotImplemented
            return f(func, params, internals)

        return jax_function_override(wrapper)

    return decorator


@jax_function.register(jax_copy)
@_jax_internals_override("a")
def protected_copy_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    if _has_numpy_protected_value(internals):
        return NotImplemented

    order = params.arguments.get("order", "C")
    return _apply_unary(internals, lambda _name, value, _axes: func(value, order=order))


@jax_function.register(jax_astype)
@_jax_internals_override("x")
def protected_astype_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    dtype = params.arguments["dtype"]
    copy = params.arguments.get("copy", True)
    device = params.arguments["device"]
    return _apply_unary(internals, lambda _name, value, _axes: func(value, dtype=dtype, copy=copy, device=device))


@jax_function.multi_register([jax_mean, jax_sum, jax_average, jax_std, jax_var])
@_jax_internals_override("a", check_is_permitted=True)
def protected_batch_reduction_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    if _has_numpy_protected_value(internals):
        return NotImplemented
    axis = params.arguments.get("axis", None)

    if params.arguments.get("returned", False):
        msg = "average with returned = True is not supported for protected-axis values."
        raise TypeError(msg)

    results: dict[str, JaxProtectedValue] = {}
    for name, axes_count in internals.protected_axes.items():
        value = internals.values[name]
        batch_ndim = value_ndim(value) - axes_count
        mapped_axis = _normalize_batch_reduction_dims(axis, batch_ndim)

        field_kwargs: dict[str, object] = {}
        for key, field_value in params.arguments.items():
            if key == "a":
                continue
            if key == "axis":
                field_kwargs[key] = mapped_axis
                continue
            field_kwargs[key] = field_value

        if func is jax_average and "weights" in field_kwargs:
            field_kwargs["weights"] = _expand_average_weights_for_protected_axes(
                field_kwargs["weights"],
                value,
                axes_count,
            )

        result = func(value, **field_kwargs)

        if axes_count == 0 and not hasattr(result, "ndim"):
            result = jnp.asarray(result)

        original_shape = value_shape(value)
        result_shape = value_shape(result)
        if protected_shape(result_shape, axes_count) != protected_shape(original_shape, axes_count):
            msg = f"Reduction modified protected trailing axes for field {name!r}."
            raise ValueError(msg)

        results[name] = cast("JaxProtectedValue", result)

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)


@jax_function.register(jax_transpose)
@_jax_internals_override("a")
def protected_transpose_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axes = params.arguments.get("axes", None)

    if axes is None:
        batch_axes = tuple(reversed(range(internals.batch_ndim)))
    else:
        if not isinstance(axes, (tuple, list)) or not all(isinstance(axis, int) for axis in axes):
            msg = "transpose axes must be a tuple/list of integers."
            raise TypeError(msg)
        batch_axes = tuple(axes)
        if len(batch_axes) != internals.batch_ndim:
            msg = "transpose axes must only refer to batch dimension."
            raise ValueError(msg)

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        full_axes = _map_batch_axes(value, axes_count, batch_axes)
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, axes=full_axes),
            lambda field_value: np.transpose(field_value, axes=full_axes),
        )

    return _apply_unary(internals, op)


@jax_function.register(jax_matrix_transpose)
@_jax_internals_override("x")
def protected_matrix_transpose_function(
    func: Callable,
    params: BoundArguments,  # noqa: ARG001
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    del func

    if internals.batch_ndim < 2:
        msg = "matrix_transpose requires at least 2 batch dimensions."
        raise ValueError(msg)

    batch_axes = list(range(internals.batch_ndim))
    batch_axes[-2], batch_axes[-1] = batch_axes[-1], batch_axes[-2]

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        full_axes = _map_batch_axes(value, axes_count, tuple(batch_axes))
        return _apply_structural_op(
            value,
            lambda field_value: jnp.transpose(cast("Any", field_value), axes=full_axes),
            lambda field_value: np.transpose(field_value, axes=full_axes),
        )

    return _apply_unary(internals, op)


@jax_function.register(jax_reshape)
@_jax_internals_override("a")
def protected_reshape_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    shape = params.arguments.get("shape", params.arguments.get("newshape", None))
    if shape is None:
        return NotImplemented

    if isinstance(shape, int):
        batch_target_shape = (shape,)
    elif isinstance(shape, (tuple, list)):
        batch_target_shape = tuple(1 if dim is None else dim for dim in shape)
    else:
        msg = "reshape shape must be an int, tuple or list."
        raise TypeError(msg)

    order = params.arguments.get("order", "C")
    copy = params.arguments.get("copy", None)

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        target_shape = (*batch_target_shape, *protected_shape(value_shape(value), axes_count))
        kwargs: dict[str, object] = {"order": order}
        if copy is not None:
            kwargs["copy"] = copy
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, target_shape, **kwargs),
            lambda field_value: np.reshape(field_value, target_shape, order=order),
        )

    return _apply_unary(internals, op)


@jax_function.register(jax_expand_dims)
@_jax_internals_override("a")
def protected_expand_dims_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis = params.arguments["axis"]
    if isinstance(axis, int):
        axis_tuple = (axis,)
    elif isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
        axis_tuple = tuple(axis)
    else:
        msg = "expand_dims axis must be an int or tuple/list of ints."
        raise TypeError(msg)

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        full_axes = normalize_axes(axis_tuple, batch_ndim, allow_endpoint=True)
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, axis=full_axes),
            lambda field_value: np.expand_dims(field_value, axis=full_axes),
        )

    return _apply_unary(internals, op)


@jax_function.register(jax_squeeze)
@_jax_internals_override("a")
def protected_squeeze_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis = params.arguments.get("axis", None)

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        shape = value_shape(value)

        if axis is None:
            squeeze_axes = tuple(i for i, size in enumerate(shape[:batch_ndim]) if size == 1)
        else:
            if isinstance(axis, int):
                axis_tuple = (axis,)
            elif isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
                axis_tuple = tuple(axis)
            else:
                msg = "squeeze axis must be an int or tuple/list of ints."
                raise TypeError(msg)

            # Explicitly requested axes are passed through unfiltered so that squeezing an axis
            # of length != 1 raises, mirroring the numpy backend.
            squeeze_axes = normalize_axes(axis_tuple, batch_ndim)

        if not squeeze_axes:
            return value

        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, axis=squeeze_axes),
            lambda field_value: np.squeeze(field_value, axis=squeeze_axes),
        )

    return _apply_unary(internals, op)


@jax_function.register(jax_swapaxes)
@_jax_internals_override("a")
def protected_swapaxes_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis1 = params.arguments["axis1"]
    axis2 = params.arguments["axis2"]
    if not isinstance(axis1, int) or not isinstance(axis2, int):
        msg = "swapaxes axis values must be integers."
        raise TypeError(msg)

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        full_axis1 = normalize_axis(axis1, batch_ndim)
        full_axis2 = normalize_axis(axis2, batch_ndim)
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, full_axis1, full_axis2),
            lambda field_value: np.swapaxes(field_value, full_axis1, full_axis2),
        )

    return _apply_unary(internals, op)


@jax_function.register(jax_moveaxis)
@_jax_internals_override("a")
def protected_moveaxis_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    source = params.arguments["source"]
    destination = params.arguments["destination"]

    if isinstance(source, int):
        source_tuple = (source,)
        source_was_int = True
    elif isinstance(source, (tuple, list)) and all(isinstance(item, int) for item in source):
        source_tuple = tuple(source)
        source_was_int = False
    else:
        msg = "moveaxis source must be an int or tuple/list of ints."
        raise TypeError(msg)

    if isinstance(destination, int):
        destination_tuple = (destination,)
        destination_was_int = True
    elif isinstance(destination, (tuple, list)) and all(isinstance(item, int) for item in destination):
        destination_tuple = tuple(destination)
        destination_was_int = False
    else:
        msg = "moveaxis destination must be an int or tuple/list of ints."
        raise TypeError(msg)

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        mapped_source = normalize_axes(source_tuple, batch_ndim)
        mapped_destination = normalize_axes(destination_tuple, batch_ndim)
        source_arg: int | tuple[int, ...] = mapped_source[0] if source_was_int else mapped_source
        destination_arg: int | tuple[int, ...] = mapped_destination[0] if destination_was_int else mapped_destination
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, source=source_arg, destination=destination_arg),
            lambda field_value: np.moveaxis(field_value, source_arg, destination_arg),
        )

    return _apply_unary(internals, op)


@jax_function.register(jax_concatenate)
@jax_function_override
def protected_concatenate_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    values = tuple(params.arguments["arrays"])
    axis = params.arguments["axis"]
    dtype = params.arguments["dtype"]

    sequence = _extract_protected_value_sequence_internals(values, None)
    template = sequence.template if sequence.template is not None else None
    if template is None:
        return NotImplemented

    if axis is not None and not isinstance(axis, int):
        msg = "concatenate axis must be an int or None."
        raise TypeError(msg)

    results: dict[str, JaxProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "concatenate with protected out requires at least one protected input."
            raise TypeError(msg)

        value = template.values[name]
        field_values = sequence.values_by_field[name]
        mapped_axis: int | None = None
        if axis is not None:
            batch_ndim = value_ndim(value) - axes_count
            mapped_axis = normalize_axis(axis, batch_ndim)
        if isinstance(value, np.ndarray):
            result = np.concatenate(
                cast("Any", field_values),
                axis=mapped_axis,
                dtype=dtype,
            )
        else:
            result = func(field_values, axis=mapped_axis, dtype=dtype)

        original_shape = value_shape(value)
        result_shape = value_shape(result)
        if protected_shape(result_shape, axes_count) != protected_shape(original_shape, axes_count):
            msg = f"Concatenation modified protected trailing axes for field {name!r}."
            raise ValueError(msg)

        results[name] = cast("JaxProtectedValue", result)

    _validate_batch_sync(results, template.protected_axes)
    return template.create(results)


@jax_function.register(jax_stack)
@jax_function_override
def protected_stack_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    values = tuple(params.arguments["arrays"])
    axis = params.arguments["axis"]
    dtype = params.arguments["dtype"]

    sequence = _extract_protected_value_sequence_internals(values, None)
    template = sequence.template if sequence.template is not None else None
    if template is None:
        return NotImplemented

    if not isinstance(axis, int):
        msg = "stack axis must be an int."
        raise TypeError(msg)

    results: dict[str, JaxProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "stack with protected out requires at least one protected input."
            raise TypeError(msg)

        field_values = sequence.values_by_field[name]
        batch_ndim = value_ndim(template.values[name]) - axes_count
        mapped_axis = normalize_axis(axis, batch_ndim, allow_endpoint=True)

        if isinstance(template.values[name], np.ndarray):
            result = np.stack(
                cast("Any", field_values),
                axis=mapped_axis,
                dtype=dtype,
            )
        else:
            result = func(field_values, axis=mapped_axis, dtype=dtype)
        results[name] = result

    _validate_batch_sync(results, template.protected_axes)
    return template.create(results)


@jax_function.register(jax_take_along_axis)
@_jax_internals_override("arr")
def protected_take_along_axis_function(
    func: Callable,
    params: BoundArguments,
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    if _has_numpy_protected_value(internals):
        return NotImplemented
    indices = params.arguments.get("indices")
    axis = params.arguments.get("axis", None)
    mode = params.arguments.get("mode")
    fill_value = params.arguments.get("fill_value")

    if not isinstance(axis, int) or not isinstance(indices, (jax.Array, np.ndarray)):
        return NotImplemented
    if indices.ndim != internals.batch_ndim:
        msg = "take_along_axis indices must have the same ndim as the protected object's batch dimensions."
        raise ValueError(msg)

    results: dict[str, JaxProtectedValue] = {}
    for name, axes_count in internals.protected_axes.items():
        value = internals.values[name]
        mapped_axis = normalize_axis(axis, value_ndim(value) - axes_count)

        field_index = indices
        for _ in range(axes_count):
            field_index = field_index[..., None]
        if axes_count > 0:
            target_shape = (*indices.shape, *protected_shape(value_shape(value), axes_count))
            field_index = jnp.broadcast_to(field_index, target_shape)

        results[name] = func(value, field_index, mapped_axis, mode=mode, fill_value=fill_value)

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)


@jax_function.register(jax_conj)
@_jax_internals_override("x")
def protected_conj_function(
    func: Callable,
    params: BoundArguments,  # noqa: ARG001
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    def op(_name: str, value: JaxProtectedValue, _axes_count: int) -> JaxProtectedValue:
        if not np.iscomplexobj(value):
            return value
        return _apply_structural_op(value, func, np.conj)

    return _apply_unary(internals, op)
