"""Shared helpers for protected-axis function dispatch implementations."""

from __future__ import annotations


def value_ndim(value: object) -> int:
    """Return ``value.ndim`` as an ``int``.

    Args:
        value: Value expected to expose an ``ndim`` attribute.

    Returns:
        The number of dimensions.

    Raises:
        TypeError: If the value does not expose ``ndim`` as an integer.
    """
    ndim = getattr(value, "ndim", None)
    if not isinstance(ndim, int):
        msg = f"Value of type {type(value).__name__} does not expose an integer ndim attribute."
        raise TypeError(msg)
    return ndim


def value_shape(value: object) -> tuple[int, ...]:
    """Return ``value.shape`` as a tuple of integers.

    Args:
        value: Value expected to expose a ``shape`` attribute.

    Returns:
        The shape tuple.

    Raises:
        TypeError: If the value does not expose an iterable integer shape.
    """
    shape = getattr(value, "shape", None)
    if shape is None:
        msg = f"Value of type {type(value).__name__} does not expose a shape attribute."
        raise TypeError(msg)

    try:
        shape_tuple = tuple(int(dim) for dim in shape)
    except TypeError as exc:
        msg = f"Value of type {type(value).__name__} does not expose an iterable integer shape."
        raise TypeError(msg) from exc

    return shape_tuple


def batch_shape(shape: tuple[int, ...], protected_axes: int) -> tuple[int, ...]:
    """Return the batch prefix of a full shape."""
    return shape if protected_axes == 0 else shape[:-protected_axes]


def protected_shape(shape: tuple[int, ...], protected_axes: int) -> tuple[int, ...]:
    """Return the protected trailing suffix of a full shape."""
    return () if protected_axes == 0 else shape[-protected_axes:]


def normalize_axis(axis: int, ndim: int, *, allow_endpoint: bool = False) -> int:
    """Normalize a possibly-negative batch axis and validate bounds."""
    bound = ndim + (1 if allow_endpoint else 0)
    normalized = axis + bound if axis < 0 else axis
    if normalized < 0 or normalized >= bound:
        msg = f"axis {axis} is out of bounds for batch dimensions with ndim {ndim}."
        raise ValueError(msg)
    return normalized


def normalize_axes(axes: tuple[int, ...], ndim: int, *, allow_endpoint: bool = False) -> tuple[int, ...]:
    """Normalize and validate a tuple of batch axes."""
    return tuple(normalize_axis(axis, ndim, allow_endpoint=allow_endpoint) for axis in axes)


def coerce_axis_tuple(axis: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Convert axis input into a tuple form."""
    return (axis,) if isinstance(axis, int) else tuple(axis)
