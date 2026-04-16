"""NumPy array-function implementations for protected-axis values."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from inspect import BoundArguments, signature
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    normalize_axes,
    normalize_axis,
    protected_shape,
    value_ndim,
    value_shape,
)
from probly.representation.array_like import NumpyArrayLike
from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable


type ArrayProtectedValue = NumpyArrayLike[Any] | np.ndarray


class ArrayAxisProtectedCreator(Protocol):
    """Protocol for rebuilding protected-axis representations."""

    def __call__(self, values: dict[str, ArrayProtectedValue]) -> Any:  # noqa: ANN401
        """Create object from updated protected values."""


@runtime_checkable
class _SupportsProtectedInternals(Protocol):
    protected_axes: dict[str, int]
    permitted_functions: set[Callable[..., Any]]

    def protected_values(self) -> dict[str, ArrayProtectedValue]:
        """Return protected field values."""

    def with_protected_values(self, values: dict[str, ArrayProtectedValue]) -> Any:  # noqa: ANN401
        """Create a copy with updated protected values."""


@dataclass(frozen=True, slots=True)
class ArrayAxisProtectedInternals:
    """Internal representation for one protected-axis object."""

    create: ArrayAxisProtectedCreator
    values: dict[str, ArrayProtectedValue]
    protected_axes: dict[str, int]
    primary_name: str
    owner_type: type[Any]
    permitted_functions: set[Callable[..., Any]]

    @property
    def primary_value(self) -> ArrayProtectedValue:
        return self.values[self.primary_name]

    @property
    def batch_ndim(self) -> int:
        axes = self.protected_axes[self.primary_name]
        return value_ndim(self.primary_value) - axes


@dataclass(frozen=True, slots=True)
class ProtectedValueSequenceInternals:
    """Extracted internals for sequence-based operations."""

    has_protected: bool
    template: ArrayAxisProtectedInternals | None
    values_by_field: dict[str, list[object]]


def array_axis_protected_internals(obj: object) -> ArrayAxisProtectedInternals | None:
    """Extract protected-axis internals from object."""
    if not isinstance(obj, _SupportsProtectedInternals):
        return None

    protected_axes = obj.protected_axes
    values = obj.protected_values()
    if not isinstance(protected_axes, dict) or len(protected_axes) == 0:
        return None

    for name, axes in protected_axes.items():
        if name not in values:
            return None
        ndim = value_ndim(values[name])
        if ndim < axes:
            return None

    primary_name = next(iter(protected_axes))
    create: ArrayAxisProtectedCreator = obj.with_protected_values
    owner_type = type(obj)
    permitted_functions = set(getattr(owner_type, "permitted_functions", set()))
    return ArrayAxisProtectedInternals(
        create=create,
        values=dict(values),
        protected_axes=dict(protected_axes),
        primary_name=primary_name,
        owner_type=owner_type,
        permitted_functions=permitted_functions,
    )


def _validate_batch_sync(values: dict[str, ArrayProtectedValue], protected_axes: dict[str, int]) -> None:
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


def _map_batch_axes(
    value: ArrayProtectedValue, protected_axes_count: int, batch_axes: tuple[int, ...]
) -> tuple[int, ...]:
    ndim = value_ndim(value)
    batch_ndim = ndim - protected_axes_count
    normalized = normalize_axes(batch_axes, batch_ndim)
    return (*normalized, *range(batch_ndim, ndim))


def _is_permitted_function(internals: ArrayAxisProtectedInternals, func: Callable) -> bool:
    return func in internals.permitted_functions


def _normalize_batch_reduction_axes(axis: object, batch_ndim: int) -> int | tuple[int, ...]:
    if axis is None:
        return tuple(range(batch_ndim))
    if isinstance(axis, int):
        return normalize_axis(axis, batch_ndim)
    if isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
        axis_tuple = cast("tuple[int, ...]", tuple(axis))
        return normalize_axes(axis_tuple, batch_ndim)

    msg = "reduction axis must be None, an int, or a tuple/list of ints."
    raise TypeError(msg)


def _apply_unary(
    internals: ArrayAxisProtectedInternals,
    op: Callable[[str, ArrayProtectedValue, int], ArrayProtectedValue],
) -> Any:  # noqa: ANN401
    results: dict[str, ArrayProtectedValue] = {}
    for name, value in internals.values.items():
        results[name] = op(name, value, internals.protected_axes[name])

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)


def _extract_protected_value_sequence_internals(values: tuple[object, ...]) -> ProtectedValueSequenceInternals:
    """Extract and align protected values for sequence operations."""
    template: ArrayAxisProtectedInternals | None = None
    values_by_field: dict[str, list[object]] = {}
    has_protected = False

    for value in values:
        internals = array_axis_protected_internals(value)
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


class _ArrayFunction(Protocol):
    def __call__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        ...


class _BoundArrayFunction(Protocol):
    def __call__(
        self,
        func: Callable,
        params: BoundArguments,
    ) -> Any:  # noqa: ANN401
        ...


class _BoundArrayFunctionWithInternals(Protocol):
    def __call__(
        self,
        func: Callable,
        params: BoundArguments,
        internals: ArrayAxisProtectedInternals,
    ) -> Any:  # noqa: ANN401
        ...


@switchdispatch
def array_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of NumPy array functions for protected-axis objects."""
    del func, args, kwargs
    return NotImplemented


def array_function_override(array_func: _BoundArrayFunction) -> _ArrayFunction:
    """Decorator converting a bound function into ``__array_function__`` shape."""

    @wraps(array_func)
    def wrapper(
        func: Callable,
        types: tuple[type[Any], ...],  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        params = signature(func).bind(*args, **kwargs)
        params.apply_defaults()
        return array_func(func, params)

    return wrapper


def array_internals_override(
    array_param_name: str,
) -> Callable[[_BoundArrayFunctionWithInternals], _ArrayFunction]:
    """Decorator for functions that operate on one protected-axis argument."""

    def decorator(f: _BoundArrayFunctionWithInternals) -> _ArrayFunction:
        @wraps(f)
        def wrapper(
            func: Callable,
            params: BoundArguments,
        ) -> Any:  # noqa: ANN401
            argument = params.arguments[array_param_name]
            internals = array_axis_protected_internals(argument)
            if internals is None:
                return NotImplemented
            return f(func, params, internals)

        return array_function_override(wrapper)

    return decorator


@array_function.register(np.copy)
@array_internals_override("a")
def protected_copy_function(
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    order = params.arguments.get("order", "C")
    subok = params.arguments.get("subok", True)

    if not subok:
        if len(internals.protected_axes) != 1:
            msg = "Cannot copy multi-field protected object with subok=False."
            raise TypeError(msg)
        return func(internals.primary_value, order=order, subok=subok)

    return _apply_unary(internals, lambda _name, value, _axes: func(value, order=order, subok=subok))


@array_function.register(np.astype)
@array_internals_override("x")
def protected_astype_function(
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    dtype = params.arguments["dtype"]
    copy = params.arguments.get("copy", True)
    return _apply_unary(internals, lambda _name, value, _axes: func(value, dtype=dtype, copy=copy))


@array_function.multi_register([np.mean, np.sum])
@array_internals_override("a")
def protected_batch_reduction_function(  # noqa: PLR0912
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    if not _is_permitted_function(internals, func):
        return NotImplemented

    axis = params.arguments.get("axis", None)
    out = params.arguments.get("out", None)
    out_internals = array_axis_protected_internals(out)
    if out_internals is not None and out_internals.protected_axes != internals.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if out is not None and out_internals is None and len(internals.protected_axes) != 1:
        msg = "non-protected out is only supported for single-field protected objects."
        raise TypeError(msg)

    results: dict[str, ArrayProtectedValue] = {}
    for name, axes_count in internals.protected_axes.items():
        value = internals.values[name]
        batch_ndim = value_ndim(value) - axes_count
        mapped_axis = _normalize_batch_reduction_axes(axis, batch_ndim)

        field_kwargs: dict[str, object] = {}
        for key, field_value in params.arguments.items():
            if key == "a":
                continue
            if key == "axis":
                field_kwargs[key] = mapped_axis
                continue
            if key == "out":
                if out is None:
                    field_kwargs[key] = None
                elif out_internals is not None:
                    field_kwargs[key] = out_internals.values[name]
                else:
                    field_kwargs[key] = out
                continue
            field_kwargs[key] = field_value

        result = func(value, **field_kwargs)

        if out is not None:
            continue

        if axes_count == 0 and not hasattr(result, "ndim"):
            result = np.asarray(result)

        original_shape = value_shape(value)
        result_shape = value_shape(result)
        if protected_shape(result_shape, axes_count) != protected_shape(original_shape, axes_count):
            msg = f"Reduction modified protected trailing axes for field {name!r}."
            raise ValueError(msg)

        results[name] = cast("ArrayProtectedValue", result)

    if out is not None:
        return out

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)


@array_function.register(np.transpose)
@array_internals_override("a")
def protected_transpose_function(
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
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
            msg = "transpose axes must only refer to batch dimensions."
            raise ValueError(msg)

    def op(_name: str, value: ArrayProtectedValue, axes_count: int) -> ArrayProtectedValue:
        full_axes = _map_batch_axes(value, axes_count, batch_axes)
        return func(value, axes=full_axes)

    return _apply_unary(internals, op)


@array_function.register(np.matrix_transpose)
@array_internals_override("x")
def protected_matrix_transpose_function(
    func: Callable,
    params: BoundArguments,  # noqa: ARG001
    internals: ArrayAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    del func

    if internals.batch_ndim < 2:
        msg = "matrix_transpose requires at least 2 batch dimensions."
        raise ValueError(msg)

    batch_axes = list(range(internals.batch_ndim))
    batch_axes[-2], batch_axes[-1] = batch_axes[-1], batch_axes[-2]

    def op(_name: str, value: ArrayProtectedValue, axes_count: int) -> ArrayProtectedValue:
        full_axes = _map_batch_axes(value, axes_count, tuple(batch_axes))
        return np.transpose(value, axes=full_axes)

    return _apply_unary(internals, op)


@array_function.register(np.reshape)
@array_internals_override("a")
def protected_reshape_function(
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    shape = params.arguments.get("shape", params.arguments.get("newshape", None))
    if shape is None:
        return NotImplemented

    if isinstance(shape, int):
        batch_target_shape = (shape,)
    else:
        if not isinstance(shape, (tuple, list)):
            msg = "reshape newshape must be an int, tuple, or list."
            raise TypeError(msg)
        batch_target_shape = tuple(1 if dim is None else dim for dim in shape)

    order = params.arguments.get("order", "C")
    copy = params.arguments.get("copy", None)

    def op(_name: str, value: ArrayProtectedValue, axes_count: int) -> ArrayProtectedValue:
        target_shape = (*batch_target_shape, *protected_shape(value_shape(value), axes_count))
        kwargs: dict[str, object] = {"order": order}
        if copy is not None:
            kwargs["copy"] = copy
        return func(value, target_shape, **kwargs)

    return _apply_unary(internals, op)


@array_function.register(np.expand_dims)
@array_internals_override("a")
def protected_expand_dims_function(
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis = params.arguments["axis"]
    if isinstance(axis, int):
        axis_tuple = (axis,)
    elif isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
        axis_tuple = tuple(axis)
    else:
        msg = "expand_dims axis must be an int or tuple/list of ints."
        raise TypeError(msg)

    def op(_name: str, value: ArrayProtectedValue, axes_count: int) -> ArrayProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        full_axes = normalize_axes(axis_tuple, batch_ndim, allow_endpoint=True)
        return func(value, axis=full_axes)

    return _apply_unary(internals, op)


@array_function.register(np.squeeze)
@array_internals_override("a")
def protected_squeeze_function(
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis = params.arguments.get("axis", None)

    def op(_name: str, value: ArrayProtectedValue, axes_count: int) -> ArrayProtectedValue:
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

            squeeze_axes = normalize_axes(axis_tuple, batch_ndim)

        return func(value, axis=squeeze_axes)

    return _apply_unary(internals, op)


@array_function.register(np.swapaxes)
@array_internals_override("a")
def protected_swapaxes_function(
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis1 = params.arguments["axis1"]
    axis2 = params.arguments["axis2"]
    if not isinstance(axis1, int) or not isinstance(axis2, int):
        msg = "swapaxes axis values must be integers."
        raise TypeError(msg)

    def op(_name: str, value: ArrayProtectedValue, axes_count: int) -> ArrayProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        full_axis1 = normalize_axis(axis1, batch_ndim)
        full_axis2 = normalize_axis(axis2, batch_ndim)
        return func(value, full_axis1, full_axis2)

    return _apply_unary(internals, op)


@array_function.register(np.moveaxis)
@array_internals_override("a")
def protected_moveaxis_function(
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
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

    def op(_name: str, value: ArrayProtectedValue, axes_count: int) -> ArrayProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        mapped_source = normalize_axes(source_tuple, batch_ndim)
        mapped_destination = normalize_axes(destination_tuple, batch_ndim)
        source_arg: int | tuple[int, ...] = mapped_source[0] if source_was_int else mapped_source
        destination_arg: int | tuple[int, ...] = mapped_destination[0] if destination_was_int else mapped_destination
        return func(value, source=source_arg, destination=destination_arg)

    return _apply_unary(internals, op)


@array_function.register(np.concatenate)
def protected_concatenate_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    values = tuple(args[0])
    axis = kwargs.get("axis", 0)
    out = kwargs.get("out")

    out_internals = array_axis_protected_internals(out)
    sequence = _extract_protected_value_sequence_internals(values)
    template = sequence.template if sequence.template is not None else out_internals
    if template is None:
        return NotImplemented

    if out_internals is not None and out_internals.protected_axes != template.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if axis is not None and not isinstance(axis, int):
        msg = "concatenate axis must be an int or None."
        raise TypeError(msg)

    results: dict[str, ArrayProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "concatenate with protected out requires at least one protected input."
            raise TypeError(msg)

        field_values = sequence.values_by_field[name]
        mapped_axis: int | None = None
        if axis is not None:
            batch_ndim = value_ndim(template.values[name]) - axes_count
            mapped_axis = normalize_axis(axis, batch_ndim)

        out_value = out_internals.values[name] if out_internals is not None else None
        result = func(field_values, axis=mapped_axis, out=out_value)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, template.protected_axes)
    return template.create(results)


@array_function.register(np.stack)
@array_function_override
def protected_stack_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    values = tuple(params.arguments["arrays"])
    axis = params.arguments.get("axis", 0)
    out = params.arguments.get("out", None)

    out_internals = array_axis_protected_internals(out)
    sequence = _extract_protected_value_sequence_internals(values)
    template = sequence.template if sequence.template is not None else out_internals
    if template is None:
        return NotImplemented

    if out_internals is not None and out_internals.protected_axes != template.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if not isinstance(axis, int):
        msg = "stack axis must be an int."
        raise TypeError(msg)

    results: dict[str, ArrayProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "stack with protected out requires at least one protected input."
            raise TypeError(msg)

        field_values = sequence.values_by_field[name]
        batch_ndim = value_ndim(template.values[name]) - axes_count
        mapped_axis = normalize_axis(axis, batch_ndim, allow_endpoint=True)

        out_value = out_internals.values[name] if out_internals is not None else None
        result = func(field_values, axis=mapped_axis, out=out_value)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, template.protected_axes)
    return template.create(results)
