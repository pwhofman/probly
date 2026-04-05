"""Shared helpers for protected-axis function dispatch implementations."""

from __future__ import annotations


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
