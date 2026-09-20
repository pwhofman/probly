"""Shared helpers for protected-axis function dispatch implementations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from inspect import BoundArguments, signature
from typing import TYPE_CHECKING, Any, Protocol, cast, overload, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class AxisProtectedCreator[V, R](Protocol):
    """Rebuild a representation from values in the same backend family."""

    def __call__(self, values: dict[str, V]) -> R:
        """Create an object from updated protected values."""
        ...


@runtime_checkable
class SupportsProtectedInternals[V, R](Protocol):
    """Structural interface for reading and rebuilding protected-axis values."""

    protected_axes: dict[str, int]
    permitted_functions: set[Callable[..., Any]]

    @overload
    def protected_values(self) -> dict[str, V]: ...

    @overload
    def protected_values(self, func: Callable) -> dict[str, V] | None: ...

    def protected_values(self, func: Callable | None = None) -> dict[str, V] | None:
        """Return protected field values, optionally in a function's context."""
        ...

    def with_protected_values(self, values: dict[str, V], func: Callable | None = None) -> R:
        """Create a copy with updated protected values."""
        ...


@dataclass(frozen=True, slots=True)
class AxisProtectedInternals[V, R]:
    """Internal representation for one protected-axis object."""

    create: AxisProtectedCreator[V, R]
    values: dict[str, V]
    protected_axes: dict[str, int]
    primary_name: str
    owner_type: type[Any]

    @property
    def primary_value(self) -> V:
        """Return the primary protected value."""
        return self.values[self.primary_name]

    @property
    def batch_ndim(self) -> int:
        """Return the number of batch dimensions of the primary value."""
        return value_ndim(self.primary_value) - self.protected_axes[self.primary_name]


@dataclass(frozen=True, slots=True)
class ProtectedValueSequenceInternals[V, R]:
    """Extracted internals for sequence-based operations."""

    has_protected: bool
    template: AxisProtectedInternals[V, R] | None
    values_by_field: dict[str, list[object]]


class InternalsExtractor[V, R](Protocol):
    """Backend adapter that supplies the protected-value type at dispatch."""

    def __call__(
        self, obj: object, func: Callable | None = None, *, check_is_permitted: bool = False
    ) -> AxisProtectedInternals[V, R] | None:
        """Extract internals when the object supports protected-axis operations."""
        ...


def extract_axis_protected_internals[V, R](
    obj: SupportsProtectedInternals[V, R],
    func: Callable | None = None,
    *,
    check_is_permitted: bool = False,
) -> AxisProtectedInternals[V, R] | None:
    """Extract internals from an object with a backend-typed interface.

    Args:
        obj: Object whose protected-value family is established by the caller.
        func: Triggering function, also passed to reconstruction.
        check_is_permitted: Whether to read values in the function's context.

    Returns:
        Extracted internals, or None for an unsupported layout or function.
    """
    protected_axes = obj.protected_axes
    if not isinstance(protected_axes, dict) or len(protected_axes) == 0:
        return None
    values = obj.protected_values(func) if check_is_permitted and func is not None else obj.protected_values()
    if values is None:
        return None

    for name, axes in protected_axes.items():
        if name not in values or value_ndim(values[name]) < axes:
            return None

    def create(values: dict[str, V]) -> R:
        return obj.with_protected_values(values, func)

    return AxisProtectedInternals(
        create=create,
        values=dict(values),
        protected_axes=dict(protected_axes),
        primary_name=next(iter(protected_axes)),
        owner_type=type(obj),
    )


def validate_batch_sync(values: Mapping[str, object], protected_axes: Mapping[str, int]) -> None:
    """Validate that results retain protected axes and share a batch shape."""
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


def apply_unary[V, R](internals: AxisProtectedInternals[V, R], op: Callable[[str, V, int], V]) -> R:
    """Transform protected fields, validate their batch shapes, and rebuild."""
    results = {name: op(name, value, internals.protected_axes[name]) for name, value in internals.values.items()}
    validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)


def extract_protected_value_sequence_internals[V, R](
    values: tuple[object, ...],
    func: Callable | None = None,
    *,
    extract: InternalsExtractor[V, R],
) -> ProtectedValueSequenceInternals[V, R]:
    """Extract and align sequence fields using a backend-typed extractor.

    Args:
        values: Protected objects and unprotected operands to align.
        func: Function context passed to the extractor.
        extract: Backend adapter that determines the protected-value family.

    Returns:
        Aligned fields and the first protected object's internals. Unprotected
        operands preceding the first protected object are skipped.

    Raises:
        ValueError: If protected inputs use different protected-axis layouts.
    """
    template: AxisProtectedInternals[V, R] | None = None
    values_by_field: dict[str, list[object]] = {}
    for value in values:
        internals = extract(value, func)
        if internals is None:
            if template is None:
                continue
            for name in template.protected_axes:
                values_by_field[name].append(value)
            continue

        if template is None:
            template = internals
            values_by_field = {name: [] for name in internals.protected_axes}
        elif internals.protected_axes != template.protected_axes:
            msg = "All protected inputs must share identical protected_axes definitions."
            raise ValueError(msg)

        for name in template.protected_axes:
            values_by_field[name].append(internals.values[name])

    return ProtectedValueSequenceInternals(template is not None, template, values_by_field)


def map_batch_axes(value: object, protected_axes_count: int, batch_axes: tuple[int, ...]) -> tuple[int, ...]:
    """Append protected trailing axes to a normalized batch permutation."""
    ndim = value_ndim(value)
    batch_ndim = ndim - protected_axes_count
    normalized = normalize_axes(batch_axes, batch_ndim)
    return (*normalized, *range(batch_ndim, ndim))


def normalize_batch_reduction_axes(
    axis: object, batch_ndim: int, *, parameter_name: str = "axis"
) -> int | tuple[int, ...]:
    """Normalize reduction axes within the batch dimensions."""
    if axis is None:
        return tuple(range(batch_ndim))
    if isinstance(axis, int):
        return normalize_axis(axis, batch_ndim)
    if isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
        return normalize_axes(cast("tuple[int, ...]", tuple(axis)), batch_ndim)

    msg = f"reduction {parameter_name} must be None, an int, or a tuple/list of ints."
    raise TypeError(msg)


def has_numpy_protected_value[V, R](internals: AxisProtectedInternals[V, R]) -> bool:
    """Return whether any protected field contains a NumPy array."""
    return any(isinstance(value, np.ndarray) for value in internals.values.values())


def apply_structural_op[V](value: V, backend_op: Callable[[V], V], numpy_op: Callable[[np.ndarray], V]) -> V:
    """Apply a structural operation while preserving NumPy-backed fields."""
    if isinstance(value, np.ndarray):
        return numpy_op(value)
    return backend_op(value)


class FunctionOverride(Protocol):
    """Signature shared by backend function-dispatch overrides."""

    def __call__(
        self, func: Callable, types: tuple[type[Any], ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:  # noqa: ANN401
        """Handle a backend function call."""
        ...


class BoundFunction(Protocol):
    """Function handler receiving signature-bound arguments."""

    def __call__(self, func: Callable, params: BoundArguments) -> Any:  # noqa: ANN401
        """Handle bound arguments."""
        ...


class BoundFunctionWithInternals[V, R](Protocol):
    """Bound function handler receiving backend-typed protected internals."""

    def __call__(self, func: Callable, params: BoundArguments, internals: AxisProtectedInternals[V, R]) -> Any:  # noqa: ANN401
        """Handle bound arguments and extracted internals."""
        ...


def function_override(handler: BoundFunction) -> FunctionOverride:
    """Adapt a signature-bound handler to backend function-dispatch shape."""

    @wraps(handler)
    def wrapper(
        func: Callable,
        types: tuple[type[Any], ...],  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        params = signature(func).bind(*args, **kwargs)
        params.apply_defaults()
        return handler(func, params)

    return wrapper


def internals_override[V, R](
    parameter_name: str,
    check_is_permitted: bool = False,
    *,
    extract: InternalsExtractor[V, R],
) -> Callable[[BoundFunctionWithInternals[V, R]], FunctionOverride]:
    """Adapt a bound handler using a backend-typed internals extractor.

    Args:
        parameter_name: Bound parameter containing the protected object.
        check_is_permitted: Whether to read values in the function's context.
        extract: Backend adapter that determines the protected-value family.

    Returns:
        A decorator that binds arguments and extracts protected internals.
    """

    def decorator(handler: BoundFunctionWithInternals[V, R]) -> FunctionOverride:
        @wraps(handler)
        def wrapper(func: Callable, params: BoundArguments) -> Any:  # noqa: ANN401
            internals = extract(params.arguments[parameter_name], func, check_is_permitted=check_is_permitted)
            if internals is None:
                return NotImplemented
            return handler(func, params, internals)

        return function_override(wrapper)

    return decorator


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
