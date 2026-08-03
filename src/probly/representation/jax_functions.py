"""Override machinery and ``jax.numpy`` mirroring wrappers.

Torch offers ``torch.overrides`` and NumPy offers ``__array_function__``, so a custom array-like
object can intercept ``torch.reshape(obj, ...)`` or ``np.reshape(obj, ...)``. JAX offers neither:
``jnp.reshape(custom_obj, ...)`` cannot be intercepted at all. ``__jax_array__`` is no longer
honored during abstractification, and registering a class as a pytree only makes ``jax.jit``,
``jax.vmap`` and ``jax.tree`` work, never ``jnp.sum(custom_obj)``.

This module therefore owns the override protocol that torch gets for free. It defines a
``__jax_function__`` hook together with :func:`has_jax_function` and :func:`handle_jax_function`
(same contract as ``torch.overrides``), plus thin wrappers that mirror the ``jax.numpy`` functions
probly needs. Call sites inside probly use ``jax_reshape(x, ...)`` where torch code would write
``torch.reshape(x, ...)``; everything downstream then behaves like the torch implementations.

The only behavior callers see differently from the torch backend: ``jnp.reshape(custom_obj, ...)``
raises, while :func:`jax_reshape` works. That is a JAX limitation, not a design choice.

The override machinery duck-types on ``__jax_function__`` being defined on the argument's *type*,
so this module never imports :mod:`probly.representation.jax_like` and the import graph stays
one-way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import jax
    from jax.sharding import Sharding
    from jax.typing import ArrayLike as JaxArrayLike, DTypeLike


def _overloaded_args(relevant_args: Iterable[object]) -> list[object]:
    """Collect the arguments whose type defines ``__jax_function__``, subclasses first."""
    overloaded: list[object] = []
    overloaded_types: list[type[Any]] = []

    for arg in relevant_args:
        arg_type = type(arg)
        if arg is None or not hasattr(arg_type, "__jax_function__") or arg_type in overloaded_types:
            continue

        # Subclasses get a chance to handle the call before their base classes, mirroring
        # the ordering rules of ``torch.overrides`` and ``__array_function__``.
        index = len(overloaded_types)
        for position, other in enumerate(overloaded_types):
            if issubclass(arg_type, other):
                index = position
                break

        overloaded_types.insert(index, arg_type)
        overloaded.insert(index, arg)

    return overloaded


def has_jax_function(relevant_args: Iterable[object]) -> bool:
    """Check whether any of the arguments overrides the jax function protocol.

    Args:
        relevant_args: The arguments to inspect. ``None`` entries are ignored.

    Returns:
        True if the type of at least one argument defines ``__jax_function__``.
    """
    return any(arg is not None and hasattr(type(arg), "__jax_function__") for arg in relevant_args)


def handle_jax_function(
    public_api: Callable[..., Any],
    relevant_args: Iterable[object],
    *args: object,
    **kwargs: object,
) -> Any:  # noqa: ANN401
    """Dispatch a call to the ``__jax_function__`` implementation of the arguments.

    Args:
        public_api: The wrapper function that was called, used as the dispatch key.
        relevant_args: The arguments that may override the call.
        *args: The positional arguments the wrapper was called with.
        **kwargs: The keyword arguments the wrapper was called with.

    Returns:
        The first result that is not ``NotImplemented``.

    Raises:
        TypeError: If no argument implements ``public_api``.
    """
    overloaded = _overloaded_args(relevant_args)
    types = tuple(type(arg) for arg in overloaded)

    for arg in overloaded:
        result = type(arg).__jax_function__(public_api, types, args, kwargs)  # ty: ignore[unresolved-attribute]
        if result is not NotImplemented:
            return result

    name = getattr(public_api, "__name__", repr(public_api))
    msg = f"no implementation found for {name} on types that implement __jax_function__: {list(types)}"
    raise TypeError(msg)


def jax_reshape(
    a: JaxArrayLike,
    /,
    shape: int | Sequence[int],
    order: str = "C",
    *,
    copy: bool | None = None,
) -> jax.Array:
    """Return a reshaped version of the array, mirroring ``jax.numpy.reshape``.

    Args:
        a: The array to reshape.
        shape: The target shape.
        order: The read/write element order.
        copy: Whether to copy the data.

    Returns:
        The reshaped array.
    """
    relevant_args = (a,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_reshape, relevant_args, a, shape, order, copy=copy)
    return jnp.reshape(a, shape, order, copy=copy)


def jax_transpose(a: JaxArrayLike, /, axes: Sequence[int] | None = None) -> jax.Array:
    """Return a transposed version of the array, mirroring ``jax.numpy.transpose``.

    Args:
        a: The array to transpose.
        axes: The permutation of axes. If omitted, the axes are reversed.

    Returns:
        The transposed array.
    """
    relevant_args = (a,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_transpose, relevant_args, a, axes)
    return jnp.transpose(a, axes)


def jax_matrix_transpose(x: JaxArrayLike, /) -> jax.Array:
    """Transpose the last two axes of the array, mirroring ``jax.numpy.matrix_transpose``.

    Args:
        x: The array to transpose.

    Returns:
        The array with its last two axes swapped.
    """
    relevant_args = (x,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_matrix_transpose, relevant_args, x)
    return jnp.matrix_transpose(x)


def jax_moveaxis(
    a: JaxArrayLike,
    /,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> jax.Array:
    """Move axes of the array to new positions, mirroring ``jax.numpy.moveaxis``.

    Args:
        a: The array to move axes of.
        source: The axes to move.
        destination: The destination positions of the moved axes.

    Returns:
        The array with the axes moved.
    """
    relevant_args = (a,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_moveaxis, relevant_args, a, source, destination)
    return jnp.moveaxis(a, source, destination)


def jax_swapaxes(a: JaxArrayLike, /, axis1: int, axis2: int) -> jax.Array:
    """Swap two axes of the array, mirroring ``jax.numpy.swapaxes``.

    Args:
        a: The array to swap axes of.
        axis1: The first axis.
        axis2: The second axis.

    Returns:
        The array with the two axes swapped.
    """
    relevant_args = (a,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_swapaxes, relevant_args, a, axis1, axis2)
    return jnp.swapaxes(a, axis1, axis2)


def jax_squeeze(a: JaxArrayLike, /, axis: int | Sequence[int] | None = None) -> jax.Array:
    """Remove axes of length one, mirroring ``jax.numpy.squeeze``.

    Args:
        a: The array to squeeze.
        axis: The axis or axes to remove. If omitted, all length-one axes are removed.

    Returns:
        The squeezed array.
    """
    relevant_args = (a,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_squeeze, relevant_args, a, axis)
    return jnp.squeeze(a, axis)


def jax_expand_dims(a: JaxArrayLike, /, axis: int | Sequence[int]) -> jax.Array:
    """Insert axes of length one, mirroring ``jax.numpy.expand_dims``.

    Args:
        a: The array to expand.
        axis: The position or positions of the new axes.

    Returns:
        The expanded array.
    """
    relevant_args = (a,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_expand_dims, relevant_args, a, axis)
    return jnp.expand_dims(a, axis)


def jax_concatenate(
    arrays: Sequence[JaxArrayLike],
    /,
    axis: int | None = 0,
    dtype: DTypeLike | None = None,
) -> jax.Array:
    """Join a sequence of arrays along an existing axis, mirroring ``jax.numpy.concatenate``.

    Args:
        arrays: The arrays to concatenate.
        axis: The axis to concatenate along. ``None`` flattens the inputs first.
        dtype: The desired data type of the result.

    Returns:
        The concatenated array.
    """
    relevant_args = tuple(arrays)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_concatenate, relevant_args, arrays, axis, dtype)
    return jnp.concatenate(arrays, axis, dtype=dtype)


def jax_stack(
    arrays: Sequence[JaxArrayLike],
    /,
    axis: int = 0,
    out: None = None,
    dtype: DTypeLike | None = None,
) -> jax.Array:
    """Join a sequence of arrays along a new axis, mirroring ``jax.numpy.stack``.

    Args:
        arrays: The arrays to stack.
        axis: The axis of the result along which the inputs are stacked.
        out: Unsupported by JAX, must be ``None``.
        dtype: The desired data type of the result.

    Returns:
        The stacked array.
    """
    relevant_args = tuple(arrays)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_stack, relevant_args, arrays, axis, out, dtype)
    return jnp.stack(arrays, axis, out, dtype=dtype)


def jax_take_along_axis(
    arr: JaxArrayLike,
    /,
    indices: JaxArrayLike,
    axis: int | None = -1,
    mode: str | None = None,
    fill_value: Any = None,  # noqa: ANN401
) -> jax.Array:
    """Take values along an axis, mirroring ``jax.numpy.take_along_axis``.

    This is the JAX analogue of ``torch.gather``.

    Args:
        arr: The array to take values from.
        indices: The indices to take.
        axis: The axis to take along. ``None`` flattens the input first.
        mode: How out-of-bounds indices are handled.
        fill_value: The value used for out-of-bounds indices in ``"fill"`` mode.

    Returns:
        The gathered array.
    """
    relevant_args = (arr, indices)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_take_along_axis, relevant_args, arr, indices, axis, mode, fill_value)
    return jnp.take_along_axis(arr, indices, axis, mode=mode, fill_value=fill_value)


def jax_conj(x: JaxArrayLike, /) -> jax.Array:
    """Return the complex conjugate of the array, mirroring ``jax.numpy.conj``.

    Args:
        x: The array to conjugate.

    Returns:
        The conjugated array.
    """
    relevant_args = (x,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_conj, relevant_args, x)
    return jnp.conj(x)


def jax_copy(a: JaxArrayLike, /, order: str | None = None) -> jax.Array:
    """Return a copy of the array, mirroring ``jax.numpy.copy``.

    Args:
        a: The array to copy.
        order: The memory layout of the copy.

    Returns:
        The copied array.
    """
    relevant_args = (a,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_copy, relevant_args, a, order)
    return jnp.copy(a, order)


def jax_astype(
    x: JaxArrayLike,
    /,
    dtype: DTypeLike | None,
    *,
    copy: bool = False,
    device: jax.Device | Sharding | None = None,
) -> jax.Array:
    """Cast the array to a new data type, mirroring ``jax.numpy.astype``.

    Args:
        x: The array to cast.
        dtype: The target data type.
        copy: Whether to always return a copy.
        device: The device the result should live on.

    Returns:
        The cast array.
    """
    relevant_args = (x,)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_astype, relevant_args, x, dtype, copy=copy, device=device)
    return jnp.astype(x, dtype, copy=copy, device=device)


def jax_mean(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    *,
    where: JaxArrayLike | None = None,
) -> jax.Array:
    """Compute the arithmetic mean, mirroring ``jax.numpy.mean``.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        dtype: The data type of the accumulator.
        out: Unsupported by JAX, must be ``None``.
        keepdims: Whether reduced axes are retained with size one.
        where: Optional mask selecting the elements to include.

    Returns:
        The mean of the array.
    """
    relevant_args = (a, where)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_mean, relevant_args, a, axis, dtype, out, keepdims, where=where)
    return jnp.mean(a, axis, dtype, out, keepdims, where=where)


def jax_sum(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    initial: JaxArrayLike | None = None,
    where: JaxArrayLike | None = None,
) -> jax.Array:
    """Compute the sum, mirroring ``jax.numpy.sum``.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        dtype: The data type of the accumulator.
        out: Unsupported by JAX, must be ``None``.
        keepdims: Whether reduced axes are retained with size one.
        initial: The starting value of the reduction.
        where: Optional mask selecting the elements to include.

    Returns:
        The sum of the array.
    """
    relevant_args = (a, where)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_sum, relevant_args, a, axis, dtype, out, keepdims, initial, where)
    return jnp.sum(a, axis, dtype, out, keepdims, initial, where)


def jax_std(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: JaxArrayLike | None = None,
) -> jax.Array:
    """Compute the standard deviation, mirroring ``jax.numpy.std``.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        dtype: The data type of the accumulator.
        out: Unsupported by JAX, must be ``None``.
        ddof: The delta degrees of freedom.
        keepdims: Whether reduced axes are retained with size one.
        where: Optional mask selecting the elements to include.

    Returns:
        The standard deviation of the array.
    """
    relevant_args = (a, where)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_std, relevant_args, a, axis, dtype, out, ddof, keepdims, where=where)
    return jnp.std(a, axis, dtype, out, ddof, keepdims, where=where)


def jax_var(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: JaxArrayLike | None = None,
) -> jax.Array:
    """Compute the variance, mirroring ``jax.numpy.var``.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        dtype: The data type of the accumulator.
        out: Unsupported by JAX, must be ``None``.
        ddof: The delta degrees of freedom.
        keepdims: Whether reduced axes are retained with size one.
        where: Optional mask selecting the elements to include.

    Returns:
        The variance of the array.
    """
    relevant_args = (a, where)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_var, relevant_args, a, axis, dtype, out, ddof, keepdims, where=where)
    return jnp.var(a, axis, dtype, out, ddof, keepdims, where=where)


def jax_average(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    weights: JaxArrayLike | None = None,
    returned: bool = False,
    keepdims: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute a possibly weighted average, mirroring ``jax.numpy.average``.

    Unlike ``torch``, ``jax.numpy`` already provides a weighted average, so this wrapper only adds
    the override check.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        weights: Optional weights, broadcastable to ``a`` or matching the reduced axis.
        returned: Whether to also return the sum of the weights.
        keepdims: Whether reduced axes are retained with size one.

    Returns:
        The weighted average, and the sum of the weights if ``returned`` is True.
    """
    relevant_args = (a, weights)
    if has_jax_function(relevant_args):
        return handle_jax_function(jax_average, relevant_args, a, axis, weights, returned, keepdims)
    return jnp.average(a, axis, weights, returned, keepdims)


__all__ = [
    "handle_jax_function",
    "has_jax_function",
    "jax_astype",
    "jax_average",
    "jax_concatenate",
    "jax_conj",
    "jax_copy",
    "jax_expand_dims",
    "jax_matrix_transpose",
    "jax_mean",
    "jax_moveaxis",
    "jax_reshape",
    "jax_squeeze",
    "jax_stack",
    "jax_std",
    "jax_sum",
    "jax_swapaxes",
    "jax_take_along_axis",
    "jax_transpose",
    "jax_var",
]
