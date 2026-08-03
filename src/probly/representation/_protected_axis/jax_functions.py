"""Jax-function implementations for protected-axis values."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, cast, overload, runtime_checkable

import numpy as np
import jax
from jax import numpy as jnp

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    coerce_axis_tuple,
    normalize_axes,
    normalize_axis,
    protected_shape,
    value_ndim,
    value_shape,
)
from probly.representation.jax_functions import (
    jax_average, jax_concatenate, jax_expand_dims, jax_matrix_transpose,
    jax_mean, jax_moveaxis, jax_reshape, jax_squeeze, jax_stack,
    jax_sum, jax_take_along_axis, jax_swapaxes,
)
from probly.representation.jax_like import JaxLike
from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable


type JaxProtectedValue = JaxLike[Any] | jax.Array | np.ndarray

class JaxAxisProtectedCreator(Protocol):
    """Protocol for rebuilding protected-axis representations."""

    def __call__(self, values: dict[str, JaxProtectedValue]) -> Any:  # noqa: ANN401
        """Create object from updated protected values."""

@runtime_checkable
class _SupportsProtectedInternals(Protocol):
    protected_axes: dict[str, int]
    permitted_functions: set[Callable[..., Any]]
    
    @overload
    def protected_values(self) -> dict[str, JaxProtectedValue]: ...
     
    @overload
    def protected_values(self, func: Callable) -> dict[str, JaxProtectedValue] | None: ...
     
    def protected_values(self, func: Callable | None = None) -> dict[str, JaxProtectedValue] | None:
        """Return protected field values."""
        
    def with_protected_values(self, values: dict[str, JaxProtectedValue], func: Callable | None = None) -> Any:  # noqa: ANN401
        """Create a copy with updated protected values."""
        
@dataclass(frozen=True, slots=True)
class JaxAxisProtectedInternals:
    """Internal representation for one protected-axis object."""

    create: JaxAxisProtectedCreator
    values: dict[str, JaxProtectedValue]
    protected_axes: dict[str, int]
    primary_name: str
    owner_type: type[Any]
    
    @property
    def primary_value(self) -> JaxProtectedValue:
        """Return the primary protected value."""
        return self.values[self.primary_name]
    
    @property
    def batch_ndim(self) -> int:
        axes = self.protected_axes[self.primary_name]
        return value_ndim(self.primary_value) - axes
    
@dataclass(frozen=True, slots=True)
class ProtectedValueSequenceInternals:
    """Extracted internals for sequence-based operations."""
    
    has_protected: bool
    template: JaxAxisProtectedInternals | None
    values_by_field: dict[str, list[object]]
    
def jax_axis_protected_internals(
    obj: object, func: Callable | None = None, *, check_is_permitted: bool = False
) -> JaxAxisProtectedInternals | None:
    """Extract protected-axis internals from object."""
    if not isinstance(obj, _SupportsProtectedInternals):
        return None
    protected_axes = obj.protected_axes
    if not isinstance(protected_axes, dict) or len(protected_axes) == 0:
        return None
    values = obj.protected_values(func if check_is_permitted else None)  # ty:ignore[invalid-argument-type]

    if values is None:
        return None

    for name, axes in protected_axes.items():
        if name not in values:
            return None
        ndim = value_ndim(values[name])
        if ndim < axes:
            return None

    primary_name = next(iter(protected_axes))

    def create(values: dict[str, JaxProtectedValue]) -> Any:  # noqa: ANN401
        return obj.with_protected_values(values, func)

    owner_type = type(obj)
    return JaxAxisProtectedInternals(
        create=create,
        values=dict(values),
        protected_axes=dict(protected_axes),
        primary_name=primary_name,
        owner_type=owner_type,
    )

def _validate_batch_sync(values: dict[str, JaxProtectedValue], protected_axes: dict[str, int]) -> None:
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

def _has_numpy_protected_value(internals: JaxAxisProtectedInternals) -> bool:
    return any(isinstance(value, np.ndarray) for value in internals.values.values())

def _apply_structural_op(
    value: JaxProtectedValue,
    jax_op: Callable[[JaxProtectedValue], object],
    numpy_op: Callable[[np.ndarray], object],
) -> JaxProtectedValue:
    if isinstance(value, np.ndarray):
        return cast("JaxProtectedValue", numpy_op(value))
    return cast("JaxProtectedValue", jax_op(value))

def _apply_unary(
    internals: JaxAxisProtectedInternals,
    op: Callable[[str, JaxProtectedValue, int], JaxProtectedValue],
) -> Any:  # noqa: ANN401
    results: dict[str, JaxProtectedValue] = {}
    for name, value in internals.values.items():
        results[name] = op(name, value, internals.protected_axes[name])

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)

def _extract_protected_value_sequence_internals(
    values: tuple[object, ...], func: Callable | None = None) -> ProtectedValueSequenceInternals:
    """Extract and align protected values for sequence operations."""
    template: JaxAxisProtectedInternals | None = None
    values_by_field: dict[str, list[object]] = {}
    has_protected = False

    for value in values:
        internals = jax_axis_protected_internals(value, func)
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


def _normalize_batch_reduction_dims(dim: object, batch_ndim: int) -> int | tuple[int, ...]:
    if dim is None:
        return tuple(range(batch_ndim))
    if isinstance(dim, int):
        return normalize_axis(dim, batch_ndim)
    if isinstance(dim, (tuple, list, jax.Array.shape)) and all(isinstance(item, int) for item in dim):
        dim_tuple = cast("tuple[int, ...]", tuple(dim))
        return normalize_axes(dim_tuple, batch_ndim)

    msg = "reduction dim must be None, an int, or a tuple/list of ints."
    raise TypeError(msg)

def _expand_average_weights_for_protected_axes(
    weights: object,
    value: JaxProtectedValue,
    axes_count: int,
) -> object:
    if not isinstance(weights, jax.Array) or axes_count == 0:
        return weights

    batch_ndim = value_ndim(value) - axes_count
    if weights.ndim == batch_ndim:
        return weights.reshape((*weights.shape, *((1,) * axes_count)))

    return weights

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

class _BoundJaxFunctionWithInternals(Protocol):
    def __call__(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        internals: JaxAxisProtectedInternals,
    ) -> Any:  # noqa: ANN401
        ...

@switchdispatch
def jax_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of jax functions for protected-axis objects."""
    del func, args, kwargs
    return NotImplemented

def jax_function_override(jax_func: _BoundJaxFunction) -> _JaxFunction:
    """Decorator to convert a bound jax function to ``__jax_function__`` shape."""

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
def _jax_internals_override(
    jax_param_name: str | None = None,
    *,
    jax_param_pos: int | None = None,
    check_is_permitted: bool = False,
) -> Callable[[_BoundJaxFunctionWithInternals], _JaxFunction]: ...

@overload
def _jax_internals_override(
    *,
    jax_param_pos: int,
    check_is_permitted: bool = False,
) -> Callable[[_BoundJaxFunctionWithInternals], _JaxFunction]: ...

def _jax_internals_override(
    jax_param_name: str | None = None,
    *,
    jax_param_pos: int | None = None,
    check_is_permitted: bool = False,
) -> Callable[[_BoundJaxFunctionWithInternals], _JaxFunction]:
    """Decorator to convert a function taking a protected-axis argument."""
    if jax_param_name is None and jax_param_pos is None:
        msg = "Either jax_param_name or jax_param_pos must be provided."
        raise ValueError(msg)
    if jax_param_name is not None and jax_param_pos is not None:
        msg = "Only one of jax_param_name or jax_param_pos can be provided."
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

            if jax_param_name is not None and jax_param_name in mutable_kwargs:
                protected_arg = mutable_kwargs[jax_param_name]
            elif jax_param_pos is not None and len(mutable_args) > jax_param_pos:
                protected_arg = mutable_args[jax_param_pos]
            else:
                return NotImplemented

            internals = jax_axis_protected_internals(
                protected_arg,
                func,
                check_is_permitted=check_is_permitted,
            )
            if internals is None:
                return NotImplemented

            return f(func, tuple(mutable_args), mutable_kwargs, internals)

        return jax_function_override(wrapper)
    
    return decorator

@jax_function.multi_register([jax_mean, jax_sum, jax_average])
@_jax_internals_override(jax_param_pos=0, check_is_permitted=True)
def protected_batch_reduction_function(  # noqa: PLR0912
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    if _has_numpy_protected_value(internals):
        return NotImplemented
    dim = args[1] if len(args) > 1 else kwargs.get("dim", kwargs.get("axis"))
    out = kwargs.get("out")
    out_internals = jax_axis_protected_internals(out)
    if out_internals is not None and out_internals.protected_axes != internals.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if out is not None and out_internals is None and len(internals.protected_axes) != 1:
        msg = "non-protected out is only supported for single-field protected objects."
        raise TypeError(msg)

    mutable_args = list(args)
    mutable_kwargs = dict(kwargs)
    results: dict[str, JaxProtectedValue] = {}

    for name, axes_count in internals.protected_axes.items():
        value = internals.values[name]
        batch_ndim = value_ndim(value) - axes_count
        mapped_dim = _normalize_batch_reduction_dims(dim, batch_ndim)

        field_args = list(mutable_args)
        field_kwargs = dict(mutable_kwargs)

        if len(field_args) == 0:
            msg = "jax reduction call is missing the input argument."
            raise TypeError(msg)
            
        field_args[0] = value
        if len(field_args) > 1:
            field_args[1] = mapped_dim
        elif "axis" in field_kwargs and "dim" not in field_kwargs:
            field_kwargs["axis"] = mapped_dim
        else:
            field_kwargs["axis"] = mapped_dim

        if out is not None:
            if out_internals is not None:
                field_kwargs["out"] = out_internals.values[name]
            else:
                field_kwargs["out"] = out

        if func is jax_average and "weights" in field_kwargs:
            field_kwargs["weights"] = _expand_average_weights_for_protected_axes(
                field_kwargs["weights"],
                value,
                axes_count,
            )

        result = func(*tuple(field_args), **field_kwargs)

        if out is not None:
            continue
            
        if axes_count == 0 and not hasattr(result, "ndim"):
            result = jnp.asarray(result)

        original_shape = value_shape(value)
        result_shape = value_shape(result)
        if protected_shape(result_shape, axes_count) != protected_shape(original_shape, axes_count):
            msg = f"Reduction modified protected trailing axes for field {name!r}."
            raise ValueError(msg)

        results[name] = cast("JaxProtectedValue", result)

    if out is not None:
        return out

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)

@jax_function.register(jax_swapaxes)
@_jax_internals_override(jax_param_pos=0)
def protected_swapaxes_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis0 = args[1] if len(args) > 1 else kwargs.get("axis0")
    axis1 = args[2] if len(args) > 2 else kwargs.get("axis1")

    if not isinstance(axis0, int) or not isinstance(axis1, int):
        return NotImplemented

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        mapped0 = normalize_axis(axis0, batch_ndim)
        mapped1 = normalize_axis(axis1, batch_ndim)
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, mapped0, mapped1),
            lambda field_value: np.swapaxes(field_value, mapped0, mapped1),
        )

    return _apply_unary(internals, op)

@jax_function.register(jax_matrix_transpose)
@_jax_internals_override(jax_param_pos=0)
def protected_matrix_transpose_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    del func, args, kwargs

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        if batch_ndim < 2:
            msg = "adjoint requires at least 2 batch dimensions."
            raise ValueError(msg)

        if isinstance(value, np.ndarray):
            return np.swapaxes(value, batch_ndim - 2, batch_ndim - 1)

        result = jnp.swapaxes(cast("Any", value), batch_ndim - 2, batch_ndim - 1)
        return jax.conj(result) if jax.iscomplexobj(result) else result

    return _apply_unary(internals, op)

@jax_function.register(jax_reshape)
@_jax_internals_override(jax_param_pos=0)
def protected_reshape_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    shape = args[1] if len(args) > 1 else kwargs.get("shape")
    if shape is None:
        return NotImplemented

    if isinstance(shape, int):
        batch_target_shape = (shape,)
    elif isinstance(shape, (tuple, list, jax.Array.shape)):
        batch_target_shape = tuple(1 if dim is None else dim for dim in shape)
    else:
        msg = "reshape shape must be an int, tuple, list, or jax.Array.shape."
        raise TypeError(msg)

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        target_shape = (*batch_target_shape, *protected_shape(value_shape(value), axes_count))
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, target_shape),
            lambda field_value: np.reshape(field_value, target_shape),
        )

    return _apply_unary(internals, op)

@jax_function.register(jax_expand_dims)
@_jax_internals_override(jax_param_pos=0)
def protected_unsqueeze_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    if not isinstance(dim, int):
        return NotImplemented

    def op(_name: str, value: JaxProtectedValue, axes_count: int) -> JaxProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        full_dim = _normalize_batch_reduction_dims(dim, batch_ndim)
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, full_dim),
            lambda field_value: np.expand_dims(field_value, axis=full_dim),
        )
    return _apply_unary(internals, op)


@jax_function.register(jax_squeeze)
@_jax_internals_override(jax_param_pos=0)
def protected_squeeze_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    axis = args[1] if len(args) > 1 else kwargs.get("axis")

    def op(_name, value, axes_count):
        batch_ndim = value_ndim(value) - axes_count
        shape = value_shape(value)

        if axis is None:
            squeeze_dims = tuple(i for i, size in enumerate(shape[:batch_ndim]) if size == 1)
        elif isinstance(axis, int):
            squeeze_dims = normalize_axes(coerce_axis_tuple(axis), batch_ndim)
        elif isinstance(axis, (tuple, list)) and all(isinstance(item, int) for item in axis):
            squeeze_dims = normalize_axes(tuple(axis), batch_ndim)
        else:
            msg = "squeeze axis must be an int or tuple/list of ints."
            raise TypeError(msg)

        squeeze_dims = tuple(sorted(set(squeeze_dims)))
        if not squeeze_dims:
            return value

        if isinstance(value, np.ndarray):
            numpy_dims = tuple(a for a in squeeze_dims if shape[a] == 1)
            return np.squeeze(value, axis=numpy_dims) if numpy_dims else value

        return cast("JaxProtectedValue", func(value, axis=squeeze_dims))

    return _apply_unary(internals, op)

@jax_function.register(jax_moveaxis)
@_jax_internals_override(jax_param_pos=0)
def protected_moveaxis_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    source = args[1] if len(args) > 1 else kwargs.get("source")
    destination = args[2] if len(args) > 2 else kwargs.get("destination")

    if not isinstance(source, int) or not isinstance(destination, int):
        return NotImplemented

    def op(_name, value, axes_count):
        batch_ndim = value_ndim(value) - axes_count
        mapped_source = normalize_axes(source, batch_ndim)
        mapped_destination = normalize_axis(destination, batch_ndim)
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, mapped_source, mapped_destination),
            lambda field_value: np.moveaxis(field_value, mapped_source, mapped_destination),
        )

    return _apply_unary(internals, op)
    

@jax_function.multi_register([jax_concatenate])
@jax_function_override
def protected_cat_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    mutable_kwargs = dict(kwargs)
    mutable_args = list(args)

    values = tuple(mutable_args[0]) if len(mutable_args) > 0 else tuple(mutable_kwargs["tensors"])
    dim = mutable_kwargs.get("dim", mutable_args[1] if len(mutable_args) > 1 else 0)
    out = mutable_kwargs.get("out")

    out_internals = jax_axis_protected_internals(out)
    sequence = _extract_protected_value_sequence_internals(values)
    template = sequence.template if sequence.template is not None else out_internals
    if template is None:
        return NotImplemented

    if out_internals is not None and out_internals.protected_axes != template.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if not isinstance(dim, int):
        return NotImplemented

    results: dict[str, JaxProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "cat with protected out requires at least one protected input."
            raise TypeError(msg)

        field_values = sequence.values_by_field[name]
        batch_ndim = value_ndim(template.values[name]) - axes_count
        mapped_dim = normalize_axes(dim, batch_ndim)

        out_value = out_internals.values[name] if out_internals is not None else None
        if isinstance(template.values[name], np.ndarray):
            result = np.concatenate(
                cast("Any", field_values),
                axis=mapped_dim,
                out=cast("Any", out_value),
            )
        else:
            result = func(field_values, axis=mapped_dim)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, template.protected_axes)


@jax_function.register(jax_take_along_axis)
@_jax_internals_override(jax_param_pos=0)
def protected_gather_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: JaxAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    if _has_numpy_protected_value(internals):
        return NotImplemented
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    index = args[2] if len(args) > 2 else kwargs.get("index")
    out = kwargs.get("out")
    sparse_grad = kwargs.get("sparse_grad", False)

    if not isinstance(dim, int) or not isinstance(index, jax.Array):
        return NotImplemented
    if index.ndim != internals.batch_ndim:
        msg = "take_along_axis index must have the same ndim as the protected object's batch dimensions."
        raise ValueError(msg)

    out_internals = jax_axis_protected_internals(out)
    if out_internals is not None and out_internals.protected_axes != internals.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)
    if out is not None and out_internals is None and len(internals.protected_axes) != 1:
        msg = "non-protected out is only supported for single-field protected objects."
        raise TypeError(msg)

    results: dict[str, JaxProtectedValue] = {}
    for name, axes_count in internals.protected_axes.items():
        value = internals.values[name]
        batch_ndim = value_ndim(value) - axes_count
        mapped_dim = normalize_axes(dim, batch_ndim)

        field_index = index
        for _ in range(axes_count):
            field_index = field_index.reshape((-1, 1))
        if axes_count > 0:
            target_shape = (*index.shape, *protected_shape(value_shape(value), axes_count))
            field_index = field_index.expand(target_shape)

        out_value = out_internals.values[name] if out_internals is not None else out
        result = func(value, field_index, mapped_dim, sparse_grad=sparse_grad)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)

@jax_function.register(jax_stack)
@jax_function_override
def protected_stack_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    mutable_kwargs = dict(kwargs)
    mutable_args = list(args)
    
    values = tuple(mutable_args[0]) if len(mutable_args) > 0 else tuple(mutable_kwargs["arrays"])
    dim = mutable_kwargs.get("dim", mutable_args[1] if len(mutable_args) > 1 else 0)
    out = mutable_kwargs.get("out")

    out_internals = jax_axis_protected_internals(out)
    sequence = _extract_protected_value_sequence_internals(values)
    template = sequence.template if sequence.template is not None else out_internals
    if template is None:
        return NotImplemented

    if out_internals is not None and out_internals.protected_axes != template.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if not isinstance(dim, int):
        return NotImplemented

    results: dict[str, JaxProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "stack with protected out requires at least one protected input."
            raise TypeError(msg)

        field_values = sequence.values_by_field[name]
        batch_ndim = value_ndim(template.values[name]) - axes_count
        mapped_dim = normalize_axes(dim, batch_ndim)

        out_value = out_internals.values[name] if out_internals is not None else None
        if isinstance(template.values[name], np.ndarray):
            result = np.stack(
                cast("Any", field_values),
                axis=mapped_dim,
                out=cast("Any", out_value),
            )
        else:
            result = func(field_values, axis=mapped_dim)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, template.protected_axes)
    return template.create(results)