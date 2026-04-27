"""Numpy array function implementations for sample arrays."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch, wraps
from inspect import BoundArguments, signature
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

import numpy as np

from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from probly.representation.array_like import NumpyArrayLike, Order


class ArraySampleCreator[D: NumpyArrayLike](Protocol):
    """Protocol for creating sample arrays."""

    def __call__(self, array: D, sample_axis: int, weights: np.ndarray | None) -> Any:  # noqa: ANN401
        """Create a sample array from a numpy array and a sample axis."""


@dataclass(frozen=True, slots=True)
class ArraySampleInternals[D: NumpyArrayLike]:
    """Internal information about a sample array."""

    create: ArraySampleCreator[D]
    array: D
    sample_axis: int
    weights: np.ndarray | None = None


@singledispatch
def array_sample_internals(_: object) -> ArraySampleInternals | None:
    """Get the sample dimension of a sample array."""
    return None


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
        create_sample: ArraySampleCreator,
        array: NumpyArrayLike,
        sample_axis: int,
        weights: np.ndarray | None,
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
    return func._implementation(*args, **kwargs)  # ty: ignore[unresolved-attribute]  # noqa: SLF001


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


@overload
def array_internals_override(
    array_sample_param_name: str,
) -> Callable[[_BoundArrayFunctionWithInternals], _ArrayFunction]: ...


@overload
def array_internals_override(
    *,
    array_sample_param_pos: int,
) -> Callable[[_BoundArrayFunctionWithInternals], _ArrayFunction]: ...


def array_internals_override(
    array_sample_param_name: str | None = None, *, array_sample_param_pos: int | None = None
) -> Callable[[_BoundArrayFunctionWithInternals], _ArrayFunction]:
    """Decorator to convert a function that takes a call with an array-sample ."""
    if array_sample_param_name is None and array_sample_param_pos is None:
        msg = "Either array_sample_param_name or array_sample_param_pos must be provided."
        raise ValueError(msg)
    if array_sample_param_name is not None and array_sample_param_pos is not None:
        msg = "Only one of array_sample_param_name or array_sample_param_pos can be provided."
        raise ValueError(msg)

    def decorator(f: _BoundArrayFunctionWithInternals) -> _ArrayFunction:
        @wraps(f)
        def wrapper(
            func: Callable,
            params: BoundArguments,
        ) -> Any:  # noqa: ANN401
            param_name = next(iter(params.arguments)) if array_sample_param_name is None else array_sample_param_name
            array_sample_arg = params.arguments[param_name]
            internals = array_sample_internals(array_sample_arg)

            if internals is None:
                return NotImplemented

            params.arguments[param_name] = internals.array

            return f(
                func,
                params,
                internals.create,
                internals.array,
                internals.sample_axis,
                internals.weights,
            )

        return array_function_override(wrapper)

    return decorator


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


@array_function.register(np.copy)
@array_internals_override("a")
def array_copy_function(
    func: Callable,
    params: BoundArguments,
    create_sample: ArraySampleCreator,
    array: NumpyArrayLike,
    sample_axis: int,
    weights: np.ndarray | None,
) -> Any:  # noqa: ANN401
    """Implementation of np.copy for sample arrays."""
    order: Order = params.arguments.get("order", "C")
    subok: bool = params.arguments.get("subok", True)

    res = func(array, order=order, subok=subok)

    if sample_axis is None or not subok:
        return res

    return create_sample(res, sample_axis=sample_axis, weights=weights)


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

    a_internals = array_sample_internals(a)
    out_internals = array_sample_internals(out)

    if a_internals is None and out_internals is None:
        return NotImplemented

    if a_internals is not None:
        params.arguments["a"] = a_internals.array

    if out_internals is not None:
        params.arguments["out"] = out_internals.array

    res = func(*params.args, **params.kwargs)

    if out_internals is not None:
        return out

    if a_internals is None:
        return res

    new_sample_axis = track_sample_axis_after_reduction(a_internals.sample_axis, a_internals.array.ndim, axis, keepdims)

    if new_sample_axis is not None:
        return a_internals.create(res, sample_axis=new_sample_axis, weights=a_internals.weights)

    return res


@array_function.register(np.transpose)
@array_internals_override("a")
def array_transpose(
    func: Callable,
    params: BoundArguments,
    create_sample: ArraySampleCreator,
    array: NumpyArrayLike,
    sample_axis: int,
    weights: np.ndarray | None,
) -> Any:  # noqa: ANN401
    """Implementation of np.transpose for sample arrays."""
    axes = params.arguments.get("axes", None)

    axes = tuple(range(array.ndim)[::-1]) if axes is None else tuple(a if a >= 0 else array.ndim + a for a in axes)

    new_sample_axis = axes.index(sample_axis)
    res = func(array, axes=axes)

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@array_function.register(np.matrix_transpose)
@array_internals_override("x")
def array_matrix_transpose(
    func: Callable,
    params: BoundArguments,  # noqa: ARG001
    create_sample: ArraySampleCreator,
    array: NumpyArrayLike,
    sample_axis: int,
    weights: np.ndarray | None,
) -> Any:  # noqa: ANN401
    """Implementation of np.matrix_transpose for sample arrays."""
    a_ndim = array.ndim

    if sample_axis == a_ndim - 1:
        new_sample_axis = a_ndim - 2
    elif sample_axis == a_ndim - 2:
        new_sample_axis = a_ndim - 1
    else:
        new_sample_axis = sample_axis

    res = func(array)

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@array_function.multi_register(
    [
        np.flip,
        np.fliplr,
        np.flipud,
        np.roll,
    ],
)
@array_internals_override(array_sample_param_pos=0)
def array_sample_axis_preserving_function(
    func: Callable,
    params: BoundArguments,
    create_sample: ArraySampleCreator,
    array: NumpyArrayLike,  # noqa: ARG001
    sample_axis: int,
    weights: np.ndarray | None,
) -> Any:  # noqa: ANN401
    """Implementation of sample-axis-preserving array functions."""
    res = func(*params.args, **params.kwargs)

    return create_sample(res, sample_axis=sample_axis, weights=weights)


@array_function.register(np.reshape)
@array_internals_override("a")
def array_reshape_function(  # noqa: PLR0912
    func: Callable,
    params: BoundArguments,
    create_sample: ArraySampleCreator,
    array: NumpyArrayLike,
    sample_axis: int,
    weights: np.ndarray | None,
) -> Any:  # noqa: ANN401
    """Implementation of np.reshape for sample arrays."""
    order: Literal["C", "F", "A"] = params.arguments.get("order", "C")

    res = func(*params.args, **params.kwargs)

    a_shape = array.shape
    res_shape = res.shape
    sample_size = a_shape[sample_axis]
    new_sample_axis: int | None = None

    if order == "A":
        order = "C" if array.flags.c_contiguous else "F"

    if order == "C":
        before_pre_size = np.prod(a_shape[:sample_axis], dtype=int)
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
        before_post_size = np.prod(a_shape[sample_axis + 1 :], dtype=int)
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

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@array_function.register(np.swapaxes)
@array_internals_override("a")
def array_swapaxes_function(
    func: Callable,
    params: BoundArguments,
    create_sample: ArraySampleCreator,
    array: NumpyArrayLike,
    sample_axis: int,
    weights: np.ndarray | None,
) -> Any:  # noqa: ANN401
    """Implementation of np.swapaxes for sample arrays."""
    axis1 = params.arguments["axis1"]
    axis2 = params.arguments["axis2"]

    a_ndim = array.ndim
    axis1 = axis1 if axis1 >= 0 else a_ndim + axis1
    axis2 = axis2 if axis2 >= 0 else a_ndim + axis2

    if sample_axis == axis1:
        new_sample_axis = axis2
    elif sample_axis == axis2:
        new_sample_axis = axis1
    else:
        new_sample_axis = sample_axis

    res = func(array, axis1, axis2)

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@array_function.register(np.expand_dims)
@array_internals_override("a")
def array_expand_dims_function(
    func: Callable,
    params: BoundArguments,
    create_sample: ArraySampleCreator,
    array: NumpyArrayLike,
    sample_axis: int,
    weights: np.ndarray | None,
) -> Any:  # noqa: ANN401
    """Implementation of np.expand_dims for sample arrays."""
    axis = params.arguments["axis"]

    a_ndim = array.ndim
    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(a if a >= 0 else a_ndim + a + 1 for a in axes)

    new_sample_axis = sample_axis

    for axis in axes:
        if axis <= new_sample_axis:
            new_sample_axis += 1

    res = func(array, axis)

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@array_function.register(np.squeeze)
@array_internals_override("a")
def array_squeeze_function(
    func: Callable,
    params: BoundArguments,
    create_sample: ArraySampleCreator,
    array: NumpyArrayLike,
    sample_axis: int,
    weights: np.ndarray | None,
) -> Any:  # noqa: ANN401
    """Implementation of np.squeeze for sample arrays."""
    axis = params.arguments.get("axis", None)
    a_ndim = array.ndim

    if axis is None:
        axes: tuple[int, ...] = tuple(i for i in range(a_ndim) if array.shape[i] == 1)
    else:
        axes = axis if isinstance(axis, tuple) else (axis,)
        axes = tuple(a if a >= 0 else a_ndim + a for a in axes)

    new_sample_axis: int | None = sample_axis

    for ax in axes:
        if ax == new_sample_axis:
            new_sample_axis = None
            break
        if ax < new_sample_axis:
            new_sample_axis -= 1

    res = func(array, axes)

    if new_sample_axis is None:
        return res

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@array_function.register(np.apply_along_axis)
@array_internals_override("arr")
def array_apply_along_axis_function(
    func: Callable,
    params: BoundArguments,
    create_sample: ArraySampleCreator,
    array: NumpyArrayLike,
    sample_axis: int,
    weights: np.ndarray | None,
) -> Any:  # noqa: ANN401
    """Implementation of np.apply_along_axis for sample arrays."""
    func1d = params.arguments["func1d"]
    axis = params.arguments["axis"]

    arr_ndim = array.ndim
    axis = axis if axis >= 0 else arr_ndim + axis

    res = func(func1d, axis, array)

    if axis == sample_axis or not isinstance(res, np.ndarray):
        return res

    new_sample_axis = sample_axis if sample_axis < axis else res.ndim - arr_ndim + sample_axis

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


def _extract_sample_array_sequence_internals(
    arrays: tuple[NumpyArrayLike, ...],
) -> tuple[list[NumpyArrayLike], list[np.ndarray | None], bool, ArraySampleCreator | None, int | None, int | None]:
    """Extract the internals of a sequence of sample arrays."""
    cast_arrays: list[NumpyArrayLike] = []
    weights: list[np.ndarray | None] = []
    has_sample_arrays = False
    sample_axes: set[int] = set()
    create_sample: ArraySampleCreator | None = None
    sample_ndim: int | None = None

    for array in arrays:
        internals = array_sample_internals(array)

        if internals is None:
            cast_arrays.append(array)
            weights.append(None)
            continue

        has_sample_arrays = True
        cast_arrays.append(internals.array)
        weights.append(internals.weights)
        sample_axes.add(internals.sample_axis)

        if create_sample is None:
            create_sample = internals.create
            sample_ndim = internals.array.ndim

    if len(sample_axes) == 1:
        return cast_arrays, weights, has_sample_arrays, create_sample, next(iter(sample_axes)), sample_ndim

    return cast_arrays, weights, has_sample_arrays, None, None, None


@array_function.register(np.concatenate)
def array_concatenate_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of np.concatenate for sample arrays."""
    arrays = tuple(args[0])
    axis = kwargs.get("axis", 0)
    out = kwargs.get("out")
    out_internals = array_sample_internals(out)

    cast_arrays, weights, has_sample_arrays, create_sample, sample_axis, sample_ndim = (
        _extract_sample_array_sequence_internals(arrays)
    )

    if not has_sample_arrays and out_internals is None:
        return NotImplemented

    if out_internals is not None:
        kwargs["out"] = out_internals.array

    res = func(cast_arrays, **kwargs)

    if out is not None:
        return out

    if create_sample is None or sample_axis is None or axis is None:
        return res

    if sample_ndim is not None:
        axis = axis if axis >= 0 else sample_ndim + axis

    if any(w is not None for w in weights):
        if axis != sample_axis:
            msg = "Weighted samples only support concatenate along the sample axis."
            raise ValueError(msg)

        weights = np.concatenate(
            [w if w is not None else np.ones(cast_arrays[i].shape[sample_axis]) for i, w in enumerate(weights)]
        )
    else:
        weights = None

    return create_sample(res, sample_axis=sample_axis, weights=weights)


@array_function.register(np.stack)
@array_function_override
def array_stack_function(
    func: Callable,
    params: BoundArguments,
) -> Any:  # noqa: ANN401
    """Implementation of np.stack for sample arrays."""
    arrays = tuple(params.arguments["arrays"])
    axis = params.arguments.get("axis", 0)
    out = params.arguments.get("out", None)
    out_internals = array_sample_internals(out)

    cast_arrays, weights, has_sample_arrays, create_sample, sample_axis, sample_ndim = (
        _extract_sample_array_sequence_internals(arrays)
    )

    if not has_sample_arrays and out_internals is None:
        return NotImplemented

    params.arguments["arrays"] = cast_arrays

    if out_internals is not None:
        params.arguments["out"] = out_internals.array

    res = func(*params.args, **params.kwargs)

    if out is not None:
        return out

    if create_sample is None or sample_axis is None:
        return res

    if any(w is not None for w in weights):
        msg = "Weighted samples do not support stack."
        raise ValueError(msg)

    input_ndim = sample_ndim
    if input_ndim is None:
        return res
    axis = axis if axis >= 0 else input_ndim + axis + 1
    new_sample_axis = sample_axis + 1 if axis <= sample_axis else sample_axis

    return create_sample(res, sample_axis=new_sample_axis, weights=None)


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
