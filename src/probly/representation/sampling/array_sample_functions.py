"""Numpy array function implementations for sample arrays."""

from __future__ import annotations

from functools import singledispatch, wraps
from inspect import BoundArguments, signature
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable


@singledispatch
def array_sample_internals(_: object) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Get the sample dimension of a sample array."""
    return None, None


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


@switchdispatch
def array_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of numpy array functions for sample arrays."""
    return func._implementation(*args, **kwargs)  # type: ignore[attr-defined]  # noqa: SLF001


def array_function_override(
    array_func: _BoundArrayFunction,
) -> _ArrayFunction:
    """Decorator to convert a bound array function to an array function."""

    @wraps(array_func)
    def wrapper(
        func: Callable,
        types: tuple[type[Any], ...],  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        sig = signature(func)
        params = sig.bind(*args, **kwargs)
        params.apply_defaults()

        return array_func(func, params)

    return wrapper


def track_sample_axis_after_reduction(
    original_sample_dim: int,
    original_ndim: int,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> int | None:
    """Track the sample axis after a reduction operation."""
    if axis is None:
        return None

    axes: tuple[int, ...] = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(a if a >= 0 else original_ndim + a for a in axes)

    if keepdims:
        return original_sample_dim

    if original_sample_dim in axes:
        return None

    new_sample_axis = original_sample_dim

    for a in axes:
        if a < original_sample_dim:
            new_sample_axis -= 1

    return new_sample_axis


@array_function.multi_register(
    [
        np.argmax,
        np.argmin,
        np.average,
        np.count_nonzero,
        np.mean,
        np.median,
        np.nanargmax,
        np.nanargmin,
        np.nanmax,
        np.nanmedian,
        np.nanmin,
        np.nanprod,
        np.nanstd,
        np.nansum,
        np.nanvar,
        np.std,
        np.var,
    ],
)
@array_function_override
def array_reduction_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    """Implementation of dimension-reducing array functions with a keepdims kwarg."""
    a = params.arguments["a"]
    out = params.arguments.get("out", None)
    axis = params.arguments.get("axis", None)
    keepdims = params.arguments.get("keepdims", False)

    #  By default, keepdims is np._NoValue, which we treat as False.
    #  Since this is not documented and subject to change, we catch all non-bool cases.
    if not isinstance(keepdims, bool):
        keepdims = False
        params.arguments["keepdims"] = keepdims

    a_array, a_sample_axis = array_sample_internals(a)
    out_array, _ = array_sample_internals(out)

    if a_array is not None:
        params.arguments["a"] = a_array

    if out_array is not None:
        params.arguments["out"] = out_array

    res = func._implementation(*params.args, **params.kwargs)  # type: ignore[attr-defined]  # noqa: SLF001

    if out_array is not None:
        return out

    if a_array is None or a_sample_axis is None:
        return res

    new_sample_axis = track_sample_axis_after_reduction(a_sample_axis, a_array.ndim, axis, keepdims)

    if isinstance(res, np.ndarray) and new_sample_axis is not None:
        return type(a)(res, sample_axis=new_sample_axis)

    return res


@array_function.register(np.transpose)
@array_function_override
def array_transpose(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    """Implementation of np.transpose for sample arrays."""
    a = params.arguments["a"]
    axes = params.arguments.get("axes", None)

    a_array, a_sample_axis = array_sample_internals(a)

    if a_array is None:
        return func._implementation(*params.args, **params.kwargs)  # type: ignore[attr-defined] # noqa: SLF001

    axes = tuple(range(a_array.ndim)[::-1]) if axes is None else tuple(a if a >= 0 else a_array.ndim + a for a in axes)

    new_sample_axis = axes.index(a_sample_axis)
    res = func._implementation(a_array, axes=axes)  # type: ignore[attr-defined]  # noqa: SLF001

    return type(a)(res, sample_axis=new_sample_axis)


@array_function.register(np.matrix_transpose)
@array_function_override
def array_matrix_transpose(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    """Implementation of np.matrix_transpose for sample arrays."""
    a = params.arguments["x"]

    a_array, a_sample_axis = array_sample_internals(a)

    if a_array is None or a_sample_axis is None:
        return func._implementation(*params.args, **params.kwargs)  # type: ignore[attr-defined] # noqa: SLF001

    a_ndim = a_array.ndim

    if a_sample_axis == a_ndim - 1:
        new_sample_axis = a_ndim - 2
    elif a_sample_axis == a_ndim - 2:
        new_sample_axis = a_ndim - 1
    else:
        new_sample_axis = a_sample_axis

    res = func._implementation(a_array)  # type: ignore[attr-defined]  # noqa: SLF001

    return type(a)(res, sample_axis=new_sample_axis)


@array_function.multi_register(
    [
        np.flip,
        np.fliplr,
        np.flipud,
        np.roll,
    ],
)
def array_sample_axis_preserving_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of sample-axis-preserving array functions."""
    a = args[0]

    if a is None:
        a = kwargs.get("a")

    a_array, a_sample_axis = array_sample_internals(a)

    res = func._implementation(*args, **kwargs)  # type: ignore[attr-defined]  # noqa: SLF001

    if a_array is not None:
        return type(a)(res, sample_axis=a_sample_axis)

    return res


@array_function.register(np.reshape)
@array_function_override
def array_reshape_function(  # noqa: C901, PLR0912
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    """Implementation of np.reshape for sample arrays."""
    a = params.arguments["a"]
    order: Literal["C", "F", "A"] = params.arguments.get("order", "C")

    a_array, a_sample_axis = array_sample_internals(a)
    res = func._implementation(*params.args, **params.kwargs)  # type: ignore[attr-defined]  # noqa: SLF001

    if a_array is None or a_sample_axis is None:
        return res

    a_shape = a_array.shape
    res_shape = res.shape
    sample_size = a_shape[a_sample_axis]
    new_sample_axis: int | None = None

    if order == "A":
        order = "C" if a_array.flags.c_contiguous else "F"

    if order == "C":
        before_pre_size = np.prod(a_shape[:a_sample_axis], dtype=int)
        after_pre_size = 1

        for i in range(len(res_shape)):
            if before_pre_size == after_pre_size:
                if res_shape[i] == sample_size:
                    new_sample_axis = i
                    break
                if res_shape[i] != 1:
                    break
            if after_pre_size > before_pre_size:
                new_sample_axis = None
                break

            after_pre_size *= res_shape[i]
    elif order == "F":
        before_post_size = np.prod(a_shape[a_sample_axis + 1 :], dtype=int)
        after_post_size = 1

        for i in range(len(res_shape) - 1, -1, -1):
            if before_post_size == after_post_size:
                if res_shape[i] == sample_size:
                    new_sample_axis = i
                    break
                if res_shape[i] != 1:
                    break
            if after_post_size > before_post_size:
                new_sample_axis = None
                break

            after_post_size *= res_shape[i]

    if new_sample_axis is None:
        return res

    return type(a)(res, sample_axis=new_sample_axis)


@array_function.register(np.swapaxes)
@array_function_override
def array_swapaxes_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    """Implementation of np.swapaxes for sample arrays."""
    a = params.arguments["a"]
    axis1 = params.arguments["axis1"]
    axis2 = params.arguments["axis2"]

    a_array, a_sample_axis = array_sample_internals(a)

    if a_array is None or a_sample_axis is None:
        return func._implementation(a, axis1, axis2)  # type: ignore[attr-defined] # noqa: SLF001

    a_ndim = a_array.ndim
    axis1 = axis1 if axis1 >= 0 else a_ndim + axis1
    axis2 = axis2 if axis2 >= 0 else a_ndim + axis2

    if a_sample_axis == axis1:
        new_sample_axis = axis2
    elif a_sample_axis == axis2:
        new_sample_axis = axis1
    else:
        new_sample_axis = a_sample_axis

    res = func._implementation(a_array, axis1, axis2)  # type: ignore[attr-defined]  # noqa: SLF001

    return type(a)(res, sample_axis=new_sample_axis)


@array_function.register(np.expand_dims)
@array_function_override
def array_expand_dims_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    """Implementation of np.expand_dims for sample arrays."""
    a = params.arguments["a"]
    axis = params.arguments["axis"]

    a_array, a_sample_axis = array_sample_internals(a)

    if a_array is None or a_sample_axis is None:
        return func._implementation(a, axis)  # type: ignore[attr-defined] # noqa: SLF001

    a_ndim = a_array.ndim
    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(a if a >= 0 else a_ndim + a + 1 for a in axes)

    new_sample_axis = a_sample_axis

    for axis in axes:
        if axis <= new_sample_axis:
            new_sample_axis += 1

    res = func._implementation(a_array, axis)  # type: ignore[attr-defined]  # noqa: SLF001

    return type(a)(res, sample_axis=new_sample_axis)


@array_function.register(np.squeeze)
@array_function_override
def array_squeeze_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    """Implementation of np.squeeze for sample arrays."""
    a = params.arguments["a"]
    axis = params.arguments.get("axis", None)

    a_array, a_sample_axis = array_sample_internals(a)

    if a_array is None or a_sample_axis is None:
        return func._implementation(a, axis)  # type: ignore[attr-defined] # noqa: SLF001

    a_ndim = a_array.ndim

    if axis is None:
        axes: tuple[int, ...] = tuple(i for i in range(a_ndim) if a_array.shape[i] == 1)
    else:
        axes = axis if isinstance(axis, tuple) else (axis,)
        axes = tuple(a if a >= 0 else a_ndim + a for a in axes)

    new_sample_axis: int | None = a_sample_axis

    for ax in axes:
        if ax == new_sample_axis:
            new_sample_axis = None
            break
        if ax < new_sample_axis:  # type: ignore[operator]
            new_sample_axis -= 1  # type: ignore[operator]

    res = func._implementation(a_array, axes)  # type: ignore[attr-defined]  # noqa: SLF001

    if new_sample_axis is None:
        return res

    return type(a)(res, sample_axis=new_sample_axis)


@array_function.register(np.apply_along_axis)
@array_function_override
def array_apply_along_axis_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    """Implementation of np.apply_along_axis for sample arrays."""
    func1d = params.arguments["func1d"]
    axis = params.arguments["axis"]
    arr = params.arguments["arr"]

    arr_array, arr_sample_axis = array_sample_internals(arr)

    if arr_array is None or arr_sample_axis is None:
        return func._implementation(func1d, axis, arr)  # type: ignore[attr-defined] # noqa: SLF001

    arr_ndim = arr_array.ndim
    axis = axis if axis >= 0 else arr_ndim + axis

    res = func._implementation(func1d, axis, arr_array)  # type: ignore[attr-defined]  # noqa: SLF001

    if axis == arr_sample_axis or not isinstance(res, np.ndarray):
        return res

    new_sample_axis = arr_sample_axis if arr_sample_axis < axis else res.ndim - arr_ndim + arr_sample_axis

    return type(arr)(res, sample_axis=new_sample_axis)


#     np.argsort,
#     np.cumprod,
#     np.cumsum,
#     np.cumulative_prod,
#     np.cumulative_sum,
#     np.diff,
#     np.gradient,
#     np.nancumprod,
#     np.nancumsum,
#     np.packbits,
#     np.sort,
#     np.trapezoid,
#     np.unwrap,
