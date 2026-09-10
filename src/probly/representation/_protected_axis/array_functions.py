"""NumPy array-function implementations for protected-axis values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from probly.representation._protected_axis._common_functions import (
    AxisProtectedCreator,
    AxisProtectedInternals,
    BoundFunctionWithInternals,
    FunctionOverride,
    ProtectedValueSequenceInternals,
    SupportsProtectedInternals,
    apply_unary as _apply_unary,
    extract_axis_protected_internals,
    extract_protected_value_sequence_internals,
    function_override as array_function_override,
    internals_override,
    map_batch_axes as _map_batch_axes,
    normalize_axes,
    normalize_axis,
    normalize_batch_reduction_axes as _normalize_batch_reduction_axes,
    protected_shape,
    validate_batch_sync as _validate_batch_sync,
    value_ndim,
    value_shape,
)
from probly.representation.array_like import NumpyArrayLike
from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable
    from inspect import BoundArguments


type ArrayProtectedValue = NumpyArrayLike[Any] | np.ndarray


type ArrayAxisProtectedCreator = AxisProtectedCreator[ArrayProtectedValue, Any]
type ArrayAxisProtectedInternals = AxisProtectedInternals[ArrayProtectedValue, Any]
type _BoundArrayFunctionWithInternals = BoundFunctionWithInternals[ArrayProtectedValue, Any]


def array_axis_protected_internals(
    obj: object, func: Callable | None = None, *, check_is_permitted: bool = False
) -> ArrayAxisProtectedInternals | None:
    """Extract protected-axis internals from object."""
    if not isinstance(obj, SupportsProtectedInternals):
        return None
    # Dispatch establishes the value family; runtime protocol checks only test members.
    return extract_axis_protected_internals(
        cast("SupportsProtectedInternals[ArrayProtectedValue, Any]", obj),
        func,
        check_is_permitted=check_is_permitted,
    )


def _extract_protected_value_sequence_internals(
    values: tuple[object, ...], func: Callable | None
) -> ProtectedValueSequenceInternals[ArrayProtectedValue, Any]:
    """Extract and align protected values for sequence operations."""
    return extract_protected_value_sequence_internals(values, func, extract=array_axis_protected_internals)


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


def array_internals_override(
    array_param_name: str,
    check_is_permitted: bool = False,
) -> Callable[[_BoundArrayFunctionWithInternals], FunctionOverride]:
    """Decorator for functions that operate on one protected-axis argument."""
    return internals_override(array_param_name, check_is_permitted, extract=array_axis_protected_internals)


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


@array_function.multi_register([np.mean, np.sum, np.average])
@array_internals_override("a", check_is_permitted=True)
def protected_batch_reduction_function(  # noqa: PLR0912
    func: Callable,
    params: BoundArguments,
    internals: ArrayAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis = params.arguments.get("axis", None)
    out = params.arguments.get("out", None)
    out_internals = array_axis_protected_internals(out, None)
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

    out_internals = array_axis_protected_internals(out, None)
    sequence = _extract_protected_value_sequence_internals(values, None)
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

    out_internals = array_axis_protected_internals(out, None)
    sequence = _extract_protected_value_sequence_internals(values, None)
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
