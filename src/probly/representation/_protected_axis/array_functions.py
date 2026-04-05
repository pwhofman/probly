"""NumPy array function implementations for protected-axis arrays."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from inspect import BoundArguments, signature
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

import numpy as np

from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable


def _normalize_axis(axis: int, ndim: int, *, allow_endpoint: bool = False) -> int:
    bound = ndim + (1 if allow_endpoint else 0)
    normalized = axis + bound if axis < 0 else axis
    if normalized < 0 or normalized >= bound:
        msg = f"axis {axis} is out of bounds for batch dimensions with ndim {ndim}."
        raise ValueError(msg)
    return normalized


def _normalize_axes(axes: tuple[int, ...], ndim: int, *, allow_endpoint: bool = False) -> tuple[int, ...]:
    return tuple(_normalize_axis(axis, ndim, allow_endpoint=allow_endpoint) for axis in axes)


def _coerce_axis_tuple(axis: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return (axis,) if isinstance(axis, int) else tuple(axis)


class ArrayAxisProtectedCreator(Protocol):
    """Protocol for creating protected-axis representations."""

    def __call__(self, array: np.ndarray) -> Any:  # noqa: ANN401
        """Create an object from a protected ndarray."""


@runtime_checkable
class _SupportsProtectedInternals(Protocol):
    protected_axes: int

    def protected_array(self) -> np.ndarray:
        """Return the ndarray with protected trailing axes."""

    def with_protected_array(self, array: np.ndarray) -> Any:  # noqa: ANN401
        """Create a new object with a replaced protected array."""


@dataclass(frozen=True, slots=True)
class ArrayAxisProtectedInternals:
    """Internal information about a protected-axis representation."""

    create: ArrayAxisProtectedCreator
    array: np.ndarray
    protected_axes: int

    @property
    def batch_ndim(self) -> int:
        return self.array.ndim - self.protected_axes

    @property
    def protected_shape(self) -> tuple[int, ...]:
        return self.array.shape[-self.protected_axes :]


def array_axis_protected_internals(obj: object) -> ArrayAxisProtectedInternals | None:
    """Get internals for protected-axis representations."""
    if not isinstance(obj, _SupportsProtectedInternals):
        return None

    array = obj.protected_array()
    protected_axes = obj.protected_axes

    if not isinstance(array, np.ndarray) or not isinstance(protected_axes, int):
        return None

    create = obj.with_protected_array
    return ArrayAxisProtectedInternals(create=create, array=array, protected_axes=protected_axes)


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
        create_protected: ArrayAxisProtectedCreator,
        array: np.ndarray,
        protected_axes: int,
    ) -> Any:  # noqa: ANN401
        ...


@switchdispatch
def array_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of NumPy array functions for protected-axis arrays."""
    del func, args, kwargs
    return NotImplemented


def array_function_override(
    array_func: _BoundArrayFunction,
) -> _ArrayFunction:
    """Decorator converting a bound function into __array_function__ shape."""

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


@overload
def array_internals_override(
    array_param_name: str,
) -> Callable[[_BoundArrayFunctionWithInternals], _ArrayFunction]: ...


@overload
def array_internals_override(
    *,
    array_param_pos: int,
) -> Callable[[_BoundArrayFunctionWithInternals], _ArrayFunction]: ...


def array_internals_override(
    array_param_name: str | None = None,
    *,
    array_param_pos: int | None = None,
) -> Callable[[_BoundArrayFunctionWithInternals], _ArrayFunction]:
    """Decorator for functions that operate on one protected-axis argument."""
    if array_param_name is None and array_param_pos is None:
        msg = "Either array_param_name or array_param_pos must be provided."
        raise ValueError(msg)
    if array_param_name is not None and array_param_pos is not None:
        msg = "Only one of array_param_name or array_param_pos can be provided."
        raise ValueError(msg)

    def decorator(f: _BoundArrayFunctionWithInternals) -> _ArrayFunction:
        @wraps(f)
        def wrapper(
            func: Callable,
            params: BoundArguments,
        ) -> Any:  # noqa: ANN401
            param_name = next(iter(params.arguments)) if array_param_name is None else array_param_name
            argument = params.arguments[param_name]
            internals = array_axis_protected_internals(argument)

            if internals is None:
                return NotImplemented

            params.arguments[param_name] = internals.array

            return f(func, params, internals.create, internals.array, internals.protected_axes)

        return array_function_override(wrapper)

    return decorator


@array_function.register(np.copy)
@array_internals_override("a")
def protected_copy_function(
    func: Callable,
    params: BoundArguments,
    create_protected: ArrayAxisProtectedCreator,
    array: np.ndarray,
    protected_axes: int,  # noqa: ARG001
) -> Any:  # noqa: ANN401
    """Implementation of np.copy for protected-axis arrays."""
    order = params.arguments.get("order", "C")
    subok = params.arguments.get("subok", True)

    res = func(array, order=order, subok=subok)

    if not subok:
        return res

    return create_protected(res)


@array_function.register(np.astype)
@array_internals_override("x")
def protected_astype_function(
    func: Callable,
    params: BoundArguments,
    create_protected: ArrayAxisProtectedCreator,
    array: np.ndarray,
    protected_axes: int,  # noqa: ARG001
) -> Any:  # noqa: ANN401
    """Implementation of np.astype for protected-axis arrays."""
    dtype = params.arguments["dtype"]
    copy = params.arguments.get("copy", True)

    return create_protected(func(array, dtype=dtype, copy=copy))


@array_function.register(np.transpose)
@array_internals_override("a")
def protected_transpose_function(
    func: Callable,
    params: BoundArguments,
    create_protected: ArrayAxisProtectedCreator,
    array: np.ndarray,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of np.transpose for protected-axis arrays."""
    axes = params.arguments.get("axes", None)
    batch_ndim = array.ndim - protected_axes

    if axes is None:
        batch_axes = tuple(reversed(range(batch_ndim)))
    else:
        if not isinstance(axes, (tuple, list)) or not all(isinstance(axis, int) for axis in axes):
            msg = "transpose axes must be a tuple/list of integers."
            raise TypeError(msg)
        batch_axes = _normalize_axes(tuple(axes), batch_ndim)
        if len(batch_axes) != batch_ndim:
            msg = "transpose axes must only refer to batch dimensions."
            raise ValueError(msg)

    full_axes = (*batch_axes, *range(batch_ndim, array.ndim))
    return create_protected(func(array, axes=full_axes))


@array_function.register(np.matrix_transpose)
@array_internals_override("x")
def protected_matrix_transpose_function(
    func: Callable,
    params: BoundArguments,  # noqa: ARG001
    create_protected: ArrayAxisProtectedCreator,
    array: np.ndarray,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of np.matrix_transpose for protected-axis arrays."""
    batch_ndim = array.ndim - protected_axes
    if batch_ndim < 2:
        msg = "matrix_transpose requires at least 2 batch dimensions."
        raise ValueError(msg)

    batch_axes = list(range(batch_ndim))
    batch_axes[-2], batch_axes[-1] = batch_axes[-1], batch_axes[-2]
    full_axes = (*batch_axes, *range(batch_ndim, array.ndim))

    return create_protected(func(array, axes=full_axes))


@array_function.register(np.reshape)
@array_internals_override("a")
def protected_reshape_function(
    func: Callable,
    params: BoundArguments,
    create_protected: ArrayAxisProtectedCreator,
    array: np.ndarray,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of np.reshape for protected-axis arrays."""
    shape = params.arguments.get("shape", params.arguments.get("newshape", None))
    if shape is None:
        return NotImplemented

    if isinstance(shape, int):
        batch_shape = (shape,)
    else:
        if not isinstance(shape, (tuple, list)):
            msg = "reshape newshape must be an int, tuple, or list."
            raise TypeError(msg)
        batch_shape = tuple(1 if dim is None else dim for dim in shape)

    protected_shape = array.shape[-protected_axes:]
    full_shape = (*batch_shape, *protected_shape)
    order = params.arguments.get("order", "C")
    copy = params.arguments.get("copy", None)
    kwargs = {"order": order}
    if copy is not None:
        kwargs["copy"] = copy

    return create_protected(func(array, full_shape, **kwargs))


@array_function.register(np.expand_dims)
@array_internals_override("a")
def protected_expand_dims_function(
    func: Callable,
    params: BoundArguments,
    create_protected: ArrayAxisProtectedCreator,
    array: np.ndarray,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of np.expand_dims for protected-axis arrays."""
    axis = params.arguments["axis"]
    batch_ndim = array.ndim - protected_axes

    if isinstance(axis, int):
        axis_tuple = (axis,)
    elif isinstance(axis, (tuple, list)) and all(isinstance(a, int) for a in axis):
        axis_tuple = tuple(axis)
    else:
        msg = "expand_dims axis must be an int or tuple/list of ints."
        raise TypeError(msg)

    full_axes = _normalize_axes(axis_tuple, batch_ndim, allow_endpoint=True)
    return create_protected(func(array, axis=full_axes))


@array_function.register(np.squeeze)
@array_internals_override("a")
def protected_squeeze_function(
    func: Callable,
    params: BoundArguments,
    create_protected: ArrayAxisProtectedCreator,
    array: np.ndarray,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of np.squeeze for protected-axis arrays."""
    axis = params.arguments.get("axis", None)
    batch_ndim = array.ndim - protected_axes

    if axis is None:
        squeeze_axes = tuple(i for i, size in enumerate(array.shape[:batch_ndim]) if size == 1)
    else:
        if isinstance(axis, int):
            axis_tuple = (axis,)
        elif isinstance(axis, (tuple, list)) and all(isinstance(a, int) for a in axis):
            axis_tuple = tuple(axis)
        else:
            msg = "squeeze axis must be an int or tuple/list of ints."
            raise TypeError(msg)

        squeeze_axes = _normalize_axes(axis_tuple, batch_ndim)

    return create_protected(func(array, axis=squeeze_axes))


@array_function.register(np.swapaxes)
@array_internals_override("a")
def protected_swapaxes_function(
    func: Callable,
    params: BoundArguments,
    create_protected: ArrayAxisProtectedCreator,
    array: np.ndarray,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of np.swapaxes for protected-axis arrays."""
    axis1 = params.arguments["axis1"]
    axis2 = params.arguments["axis2"]
    if not isinstance(axis1, int) or not isinstance(axis2, int):
        msg = "swapaxes axis values must be integers."
        raise TypeError(msg)

    batch_ndim = array.ndim - protected_axes
    full_axis1 = _normalize_axis(axis1, batch_ndim)
    full_axis2 = _normalize_axis(axis2, batch_ndim)
    return create_protected(func(array, full_axis1, full_axis2))


@array_function.register(np.moveaxis)
@array_internals_override("a")
def protected_moveaxis_function(
    func: Callable,
    params: BoundArguments,
    create_protected: ArrayAxisProtectedCreator,
    array: np.ndarray,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of np.moveaxis for protected-axis arrays."""
    source = params.arguments["source"]
    destination = params.arguments["destination"]
    batch_ndim = array.ndim - protected_axes

    if isinstance(source, int):
        source_tuple = (source,)
        source_was_int = True
    elif isinstance(source, (tuple, list)) and all(isinstance(s, int) for s in source):
        source_tuple = tuple(source)
        source_was_int = False
    else:
        msg = "moveaxis source must be an int or tuple/list of ints."
        raise TypeError(msg)

    if isinstance(destination, int):
        destination_tuple = (destination,)
        destination_was_int = True
    elif isinstance(destination, (tuple, list)) and all(isinstance(d, int) for d in destination):
        destination_tuple = tuple(destination)
        destination_was_int = False
    else:
        msg = "moveaxis destination must be an int or tuple/list of ints."
        raise TypeError(msg)

    mapped_source = _normalize_axes(source_tuple, batch_ndim)
    mapped_destination = _normalize_axes(destination_tuple, batch_ndim)

    source_arg: int | tuple[int, ...] = mapped_source[0] if source_was_int else mapped_source
    destination_arg: int | tuple[int, ...] = mapped_destination[0] if destination_was_int else mapped_destination

    return create_protected(func(array, source=source_arg, destination=destination_arg))
