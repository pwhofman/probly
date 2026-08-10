"""JAX function implementations for sample arrays."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch, wraps
import math
from typing import TYPE_CHECKING, Any, Protocol, overload

import jax.numpy as jnp

from probly.representation.jax_functions import (
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
from probly.representation.sample.array_functions import track_sample_axis_after_reduction
from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import jax

    from probly.representation.jax_like import JaxLike


class JaxSampleCreator[D: JaxLike | jax.Array](Protocol):
    """Protocol for creating sample arrays."""

    def __call__(self, array: D, sample_axis: int, weights: jax.Array | None) -> Any:  # noqa: ANN401
        """Create a sample array from a JAX array and a sample axis."""


@dataclass(frozen=True, slots=True)
class JaxSampleInternals[D: JaxLike | jax.Array]:
    """Internal information about a sample array."""

    create: JaxSampleCreator[D]
    array: D
    sample_axis: int
    weights: jax.Array | None = None


@singledispatch
def jax_sample_internals(_: object) -> JaxSampleInternals | None:
    """Get internals for a sample array."""
    return None


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
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        ...


class _BoundJaxFunctionWithInternals[D: JaxLike | jax.Array](Protocol):
    def __call__(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        create_sample: JaxSampleCreator[D],
        array: JaxLike[D],
        sample_axis: int,
        weights: jax.Array | None,
    ) -> Any:  # noqa: ANN401
        ...


@switchdispatch
def jax_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of the jax function wrappers for sample arrays."""
    del func, args, kwargs
    return NotImplemented


def jax_function_override(
    jax_func: _BoundJaxFunction,
) -> _JaxFunction:
    """Decorator to convert a bound jax function to a jax function."""

    @wraps(jax_func)
    def wrapper(
        func: Callable,
        types: tuple[type[Any], ...],  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        return jax_func(func, args, kwargs)

    return wrapper


@overload
def jax_internals_override(
    jax_sample_param_name: str,
) -> Callable[[_BoundJaxFunctionWithInternals], _JaxFunction]: ...


@overload
def jax_internals_override(
    *,
    jax_sample_param_pos: int,
) -> Callable[[_BoundJaxFunctionWithInternals], _JaxFunction]: ...


def jax_internals_override(
    jax_sample_param_name: str | None = None, *, jax_sample_param_pos: int | None = None
) -> Callable[[_BoundJaxFunctionWithInternals], _JaxFunction]:
    """Decorator to convert a function taking a sample array argument."""
    if jax_sample_param_name is None and jax_sample_param_pos is None:
        msg = "Either jax_sample_param_name or jax_sample_param_pos must be provided."
        raise ValueError(msg)
    if jax_sample_param_name is not None and jax_sample_param_pos is not None:
        msg = "Only one of jax_sample_param_name or jax_sample_param_pos can be provided."
        raise ValueError(msg)

    def decorator(f: _BoundJaxFunctionWithInternals) -> _JaxFunction:
        @wraps(f)
        def wrapper(
            func: Callable,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:  # noqa: ANN401
            mutable_kwargs = dict(kwargs)
            mutable_args = list(args)
            in_kwargs: bool
            param_name: str
            param_pos: int

            if jax_sample_param_name is not None and jax_sample_param_name in mutable_kwargs:
                sample_arg = mutable_kwargs[jax_sample_param_name]
                in_kwargs = True
                param_name = jax_sample_param_name
            elif jax_sample_param_pos is not None and len(mutable_args) > jax_sample_param_pos:
                sample_arg = mutable_args[jax_sample_param_pos]
                in_kwargs = False
                param_pos = jax_sample_param_pos
            else:
                return NotImplemented

            internals = jax_sample_internals(sample_arg)

            if internals is None:
                return NotImplemented

            if in_kwargs:
                mutable_kwargs[param_name] = internals.array
            else:
                mutable_args[param_pos] = internals.array

            return f(
                func,
                tuple(mutable_args),
                mutable_kwargs,
                internals.create,
                internals.array,
                internals.sample_axis,
                internals.weights,
            )

        return jax_function_override(wrapper)

    return decorator


def _extract_sample_array_sequence_internals(
    arrays: tuple[JaxLike, ...],
) -> tuple[list[JaxLike], list[jax.Array | None], bool, JaxSampleCreator | None, int | None, int | None]:
    """Extract internals from a sequence of arrays or sample arrays."""
    cast_arrays: list[JaxLike] = []
    weights: list[jax.Array | None] = []
    has_sample_arrays = False
    sample_axes: set[int] = set()
    create_sample: JaxSampleCreator | None = None
    sample_ndim: int | None = None

    for array in arrays:
        internals = jax_sample_internals(array)

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


def _normalize_axis(axis: object) -> Any:  # noqa: ANN401
    """Bring the axis argument of a reduction into the form the axis tracker expects.

    Args:
        axis: The axis argument as it was passed to the wrapper.

    Returns:
        ``None``, an int or a tuple of ints, or ``NotImplemented`` for unsupported inputs.
    """
    if axis is None or (isinstance(axis, int) and not isinstance(axis, bool)):
        return axis
    if isinstance(axis, (tuple, list)) and all(isinstance(a, int) and not isinstance(a, bool) for a in axis):
        return tuple(axis)
    return NotImplemented


def _normalize_axes_sequence(axes: object, ndim: int) -> tuple[int, ...] | None:
    """Normalize an int or a sequence of ints into a tuple of non-negative axes.

    Args:
        axes: The axis or axes to normalize.
        ndim: The number of dimensions the axes refer to.

    Returns:
        The normalized axes, or ``None`` if the input is not a valid axis specification.
    """
    if isinstance(axes, bool):
        return None
    if isinstance(axes, int):
        candidates: Sequence[Any] = (axes,)
    elif isinstance(axes, (tuple, list)):
        candidates = axes
    else:
        return None

    normalized: list[int] = []
    for axis in candidates:
        if isinstance(axis, bool) or not isinstance(axis, int):
            return None
        resolved = axis if axis >= 0 else ndim + axis
        if resolved < 0 or resolved >= ndim:
            return None
        normalized.append(resolved)

    return tuple(normalized)


def _track_sample_axis_after_reshape(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    sample_axis: int,
    order: str,
) -> int | None:
    """Track the sample axis through a reshape, mirroring the numpy backend.

    Args:
        in_shape: The shape of the array before the reshape.
        out_shape: The shape of the array after the reshape.
        sample_axis: The sample axis before the reshape.
        order: The read/write element order, either ``"C"`` or ``"F"``.

    Returns:
        The sample axis of the reshaped array, or ``None`` if it does not survive.
    """
    sample_size = in_shape[sample_axis]

    if order == "C":
        before_size = math.prod(in_shape[:sample_axis])
        positions: range = range(len(out_shape))
    else:
        before_size = math.prod(in_shape[sample_axis + 1 :])
        positions = range(len(out_shape) - 1, -1, -1)

    after_size = 1

    for i in positions:
        if before_size == after_size:
            if out_shape[i] == sample_size:
                return i
            if out_shape[i] != 1:
                return None
        if after_size > before_size:
            return None

        after_size *= out_shape[i]

    return None


#: Position of the ``keepdims`` parameter in each reduction wrapper's signature.
_REDUCTION_KEEPDIMS_POSITION: dict[Callable, int] = {
    jax_average: 4,
    jax_mean: 4,
    jax_std: 5,
    jax_sum: 4,
    jax_var: 5,
}


@jax_function.multi_register(
    [
        jax_average,
        jax_mean,
        jax_std,
        jax_sum,
        jax_var,
    ],
)
@jax_function_override
def jax_reduction_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of axis-reducing jax functions with a keepdims parameter.

    Unlike the torch mirror there is no ``out`` handling: ``jax.numpy`` reductions reject any
    non-``None`` ``out`` argument, so a sample array can never be an output buffer.

    Mirroring the numpy backend, the sample weights are never injected into ``jax_average``:
    they only apply to a reduction over the sample axis, which the caller expresses through
    :meth:`~probly.representation.sample.jax.JaxArraySample.sample_mean` instead.
    """
    if len(args) == 0:
        return NotImplemented

    internals = jax_sample_internals(args[0])

    if internals is None:
        return NotImplemented

    mutable_args = list(args)
    mutable_kwargs = dict(kwargs)
    mutable_args[0] = internals.array

    axis = _normalize_axis(mutable_args[1] if len(mutable_args) > 1 else mutable_kwargs.get("axis"))

    keepdims_position = _REDUCTION_KEEPDIMS_POSITION[func]
    if len(mutable_args) > keepdims_position:
        keepdims = mutable_args[keepdims_position]
    else:
        keepdims = mutable_kwargs.get("keepdims", False)

    if axis is NotImplemented or not isinstance(keepdims, bool):
        return NotImplemented

    res = func(*tuple(mutable_args), **mutable_kwargs)

    if isinstance(res, tuple):
        # ``jax_average(..., returned=True)`` returns the average and the sum of the weights.
        return res

    new_sample_axis = track_sample_axis_after_reduction(
        internals.sample_axis,
        internals.array.ndim,
        axis,
        keepdims,
    )

    if new_sample_axis is None:
        return res

    return internals.create(res, sample_axis=new_sample_axis, weights=internals.weights)


@jax_function.register(jax_transpose)
@jax_internals_override(jax_sample_param_pos=0)
def jax_transpose_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the jax transpose wrapper for sample arrays."""
    axes = args[1] if len(args) > 1 else kwargs.get("axes")

    if axes is None:
        normalized_axes = tuple(range(array.ndim))[::-1]
    elif isinstance(axes, (tuple, list)):
        normalized_axes = tuple(axis if axis >= 0 else array.ndim + axis for axis in axes)
    else:
        return NotImplemented

    new_sample_axis = normalized_axes.index(sample_axis)
    res = func(*args, **kwargs)

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@jax_function.register(jax_matrix_transpose)
@jax_internals_override(jax_sample_param_pos=0)
def jax_matrix_transpose_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the jax matrix transpose wrapper for sample arrays."""
    a_ndim = array.ndim

    if sample_axis == a_ndim - 1:
        new_sample_axis = a_ndim - 2
    elif sample_axis == a_ndim - 2:
        new_sample_axis = a_ndim - 1
    else:
        new_sample_axis = sample_axis

    res = func(*args, **kwargs)

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@jax_function.register(jax_moveaxis)
@jax_internals_override(jax_sample_param_pos=0)
def jax_moveaxis_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the jax moveaxis wrapper for sample arrays."""
    source = args[1] if len(args) > 1 else kwargs.get("source")
    destination = args[2] if len(args) > 2 else kwargs.get("destination")

    ndim = array.ndim
    normalized_source = _normalize_axes_sequence(source, ndim)
    normalized_destination = _normalize_axes_sequence(destination, ndim)

    if normalized_source is None or normalized_destination is None:
        return NotImplemented
    if len(normalized_source) != len(normalized_destination):
        return NotImplemented

    order = [axis for axis in range(ndim) if axis not in set(normalized_source)]
    for destination_axis, source_axis in sorted(zip(normalized_destination, normalized_source, strict=True)):
        order.insert(destination_axis, source_axis)

    new_sample_axis = order.index(sample_axis)
    res = func(*args, **kwargs)

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@jax_function.register(jax_conj)
@jax_internals_override(jax_sample_param_pos=0)
def jax_conj_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,  # noqa: ARG001
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the sample-axis-preserving jax conjugate wrapper for sample arrays."""
    res = func(*args, **kwargs)

    return create_sample(res, sample_axis=sample_axis, weights=weights)


@jax_function.register(jax_copy)
@jax_internals_override(jax_sample_param_pos=0)
def jax_copy_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,  # noqa: ARG001
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the jax copy wrapper for sample arrays."""
    res = func(*args, **kwargs)

    return create_sample(res, sample_axis=sample_axis, weights=weights)


@jax_function.register(jax_reshape)
@jax_internals_override(jax_sample_param_pos=0)
def jax_reshape_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the jax reshape wrapper for sample arrays.

    The sample axis survives a reshape if it maps onto exactly one axis of the result; this
    mirrors the numpy backend. Jax arrays are always C-contiguous, so ``order="A"`` is
    equivalent to ``order="C"``.
    """
    order = args[2] if len(args) > 2 else kwargs.get("order", "C")

    res = func(*args, **kwargs)

    if order == "A":
        order = "C"
    if order not in ("C", "F"):
        return res

    new_sample_axis = _track_sample_axis_after_reshape(array.shape, res.shape, sample_axis, order)

    if new_sample_axis is None:
        return res

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@jax_function.register(jax_squeeze)
@jax_internals_override(jax_sample_param_pos=0)
def jax_squeeze_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the jax squeeze wrapper for sample arrays."""
    axis = args[1] if len(args) > 1 else kwargs.get("axis")

    ndim = array.ndim

    if axis is None:
        axes: tuple[int, ...] = tuple(i for i in range(ndim) if array.shape[i] == 1)
    else:
        normalized_axes = _normalize_axes_sequence(axis, ndim)
        if normalized_axes is None:
            return NotImplemented
        axes = normalized_axes

    new_sample_axis: int | None = sample_axis

    for squeezed_axis in axes:
        if squeezed_axis == new_sample_axis:
            new_sample_axis = None
            break
        if squeezed_axis < new_sample_axis:
            new_sample_axis -= 1

    res = func(*args, **kwargs)

    if new_sample_axis is None:
        return res

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@jax_function.register(jax_expand_dims)
@jax_internals_override(jax_sample_param_pos=0)
def jax_expand_dims_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the jax expand_dims wrapper for sample arrays."""
    axis = args[1] if len(args) > 1 else kwargs.get("axis")

    ndim = array.ndim
    raw_axes: Sequence[Any] = axis if isinstance(axis, (tuple, list)) else (axis,)
    normalized_axes: list[int] = []

    for raw_axis in raw_axes:
        if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
            return NotImplemented
        normalized_axes.append(raw_axis if raw_axis >= 0 else ndim + raw_axis + 1)

    new_sample_axis = sample_axis

    for inserted_axis in normalized_axes:
        if inserted_axis <= new_sample_axis:
            new_sample_axis += 1

    res = func(*args, **kwargs)

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@jax_function.register(jax_swapaxes)
@jax_internals_override(jax_sample_param_pos=0)
def jax_swapaxes_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the jax swapaxes wrapper for sample arrays."""
    axis1 = args[1] if len(args) > 1 else kwargs.get("axis1")
    axis2 = args[2] if len(args) > 2 else kwargs.get("axis2")

    ndim = array.ndim
    normalized = _normalize_axes_sequence([axis1, axis2], ndim)

    if normalized is None:
        return NotImplemented

    first, second = normalized

    if sample_axis == first:
        new_sample_axis = second
    elif sample_axis == second:
        new_sample_axis = first
    else:
        new_sample_axis = sample_axis

    res = func(*args, **kwargs)

    return create_sample(res, sample_axis=new_sample_axis, weights=weights)


@jax_function.register(jax_take_along_axis)
@jax_internals_override(jax_sample_param_pos=0)
def jax_take_along_axis_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: JaxSampleCreator,
    array: JaxLike,
    sample_axis: int,
    weights: jax.Array | None,
) -> Any:  # noqa: ANN401
    """Implementation of the jax take_along_axis wrapper for sample arrays.

    The sample axis only survives if the gather runs along a different axis and leaves the
    number of samples untouched.
    """
    axis = args[2] if len(args) > 2 else kwargs.get("axis", -1)

    res = func(*args, **kwargs)

    if isinstance(axis, bool) or not isinstance(axis, int):
        return res

    ndim = array.ndim
    normalized_axis = axis if axis >= 0 else ndim + axis

    if normalized_axis == sample_axis or res.ndim != ndim or res.shape[sample_axis] != array.shape[sample_axis]:
        return res

    return create_sample(res, sample_axis=sample_axis, weights=weights)


@jax_function.register(jax_concatenate)
@jax_function_override
def jax_concatenate_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of the jax concatenate wrapper for sample arrays."""
    mutable_args = list(args)
    mutable_kwargs = dict(kwargs)

    if len(mutable_args) > 0:
        arrays = tuple(mutable_args[0])
        arrays_location: int | str = 0
    elif "arrays" in mutable_kwargs:
        arrays = tuple(mutable_kwargs["arrays"])
        arrays_location = "arrays"
    else:
        return NotImplemented

    axis = mutable_args[1] if len(mutable_args) > 1 else mutable_kwargs.get("axis", 0)

    if axis is not None and not isinstance(axis, int):
        return NotImplemented

    cast_arrays, weights, has_sample_arrays, create_sample, sample_axis, sample_ndim = (
        _extract_sample_array_sequence_internals(arrays)
    )

    if not has_sample_arrays:
        return NotImplemented

    if isinstance(arrays_location, int):
        mutable_args[arrays_location] = cast_arrays
    else:
        mutable_kwargs[arrays_location] = cast_arrays

    res = func(*tuple(mutable_args), **mutable_kwargs)

    if create_sample is None or sample_axis is None or axis is None:
        return res

    if sample_ndim is not None:
        axis = axis if axis >= 0 else sample_ndim + axis

    if any(w is not None for w in weights):
        if axis != sample_axis:
            msg = "Weighted samples only support concatenate along the sample axis."
            raise ValueError(msg)

        new_weights = jnp.concatenate(
            [w if w is not None else jnp.ones(cast_arrays[i].shape[sample_axis]) for i, w in enumerate(weights)]
        )
    else:
        new_weights = None

    return create_sample(res, sample_axis=sample_axis, weights=new_weights)


@jax_function.register(jax_stack)
@jax_function_override
def jax_stack_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of the jax stack wrapper for sample arrays."""
    mutable_args = list(args)
    mutable_kwargs = dict(kwargs)

    if len(mutable_args) > 0:
        arrays = tuple(mutable_args[0])
        arrays_location: int | str = 0
    elif "arrays" in mutable_kwargs:
        arrays = tuple(mutable_kwargs["arrays"])
        arrays_location = "arrays"
    else:
        return NotImplemented

    axis = mutable_args[1] if len(mutable_args) > 1 else mutable_kwargs.get("axis", 0)

    if not isinstance(axis, int):
        return NotImplemented

    cast_arrays, weights, has_sample_arrays, create_sample, sample_axis, sample_ndim = (
        _extract_sample_array_sequence_internals(arrays)
    )

    if not has_sample_arrays:
        return NotImplemented

    if isinstance(arrays_location, int):
        mutable_args[arrays_location] = cast_arrays
    else:
        mutable_kwargs[arrays_location] = cast_arrays

    res = func(*tuple(mutable_args), **mutable_kwargs)

    if create_sample is None or sample_axis is None:
        return res

    if any(w is not None for w in weights):
        msg = "Weighted samples do not support stack."
        raise ValueError(msg)

    if sample_ndim is None:
        return res

    normalized_axis = axis if axis >= 0 else sample_ndim + axis + 1
    new_sample_axis = sample_axis + 1 if normalized_axis <= sample_axis else sample_axis

    return create_sample(res, sample_axis=new_sample_axis, weights=None)


__all__ = [
    "JaxSampleCreator",
    "JaxSampleInternals",
    "jax_function",
    "jax_function_override",
    "jax_internals_override",
    "jax_sample_internals",
]
