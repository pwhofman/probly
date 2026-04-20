"""Torch-function implementations for protected-axis values."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, cast, overload, runtime_checkable

import torch

from probly.representation._protected_axis._common_functions import (
    batch_shape,
    coerce_axis_tuple,
    normalize_axes,
    normalize_axis,
    protected_shape,
    value_ndim,
    value_shape,
)
from probly.representation.torch_like import TorchLike
from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable


type TorchProtectedValue = TorchLike[Any] | torch.Tensor


class TorchAxisProtectedCreator(Protocol):
    """Protocol for rebuilding protected-axis representations."""

    def __call__(self, values: dict[str, TorchProtectedValue]) -> Any:  # noqa: ANN401
        """Create object from updated protected values."""


@runtime_checkable
class _SupportsProtectedInternals(Protocol):
    protected_axes: dict[str, int]
    permitted_functions: set[Callable[..., Any]]

    def protected_values(self) -> dict[str, TorchProtectedValue]:
        """Return protected field values."""

    def with_protected_values(self, values: dict[str, TorchProtectedValue]) -> Any:  # noqa: ANN401
        """Create a copy with updated protected values."""


@dataclass(frozen=True, slots=True)
class TorchAxisProtectedInternals:
    """Internal representation for one protected-axis object."""

    create: TorchAxisProtectedCreator
    values: dict[str, TorchProtectedValue]
    protected_axes: dict[str, int]
    primary_name: str
    owner_type: type[Any]
    permitted_functions: set[Callable[..., Any]]

    @property
    def primary_value(self) -> TorchProtectedValue:
        return self.values[self.primary_name]

    @property
    def batch_ndim(self) -> int:
        axes = self.protected_axes[self.primary_name]
        return value_ndim(self.primary_value) - axes


@dataclass(frozen=True, slots=True)
class ProtectedValueSequenceInternals:
    """Extracted internals for sequence-based operations."""

    has_protected: bool
    template: TorchAxisProtectedInternals | None
    values_by_field: dict[str, list[object]]


def torch_axis_protected_internals(obj: object) -> TorchAxisProtectedInternals | None:
    """Extract protected-axis internals from object."""
    if not isinstance(obj, _SupportsProtectedInternals):
        return None

    protected_axes = obj.protected_axes
    values = obj.protected_values()
    if not isinstance(protected_axes, dict) or len(protected_axes) == 0:
        return None

    for name, axes in protected_axes.items():
        if name not in values:
            return None
        ndim = value_ndim(values[name])
        if ndim < axes:
            return None

    primary_name = next(iter(protected_axes))
    create: TorchAxisProtectedCreator = obj.with_protected_values
    owner_type = type(obj)
    permitted_functions = set(getattr(owner_type, "permitted_functions", set()))
    return TorchAxisProtectedInternals(
        create=create,
        values=dict(values),
        protected_axes=dict(protected_axes),
        primary_name=primary_name,
        owner_type=owner_type,
        permitted_functions=permitted_functions,
    )


def _validate_batch_sync(values: dict[str, TorchProtectedValue], protected_axes: dict[str, int]) -> None:
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


def _apply_unary(
    internals: TorchAxisProtectedInternals,
    op: Callable[[str, TorchProtectedValue, int], TorchProtectedValue],
) -> Any:  # noqa: ANN401
    results: dict[str, TorchProtectedValue] = {}
    for name, value in internals.values.items():
        results[name] = op(name, value, internals.protected_axes[name])

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)


def _extract_protected_value_sequence_internals(values: tuple[object, ...]) -> ProtectedValueSequenceInternals:
    """Extract and align protected values for sequence operations."""
    template: TorchAxisProtectedInternals | None = None
    values_by_field: dict[str, list[object]] = {}
    has_protected = False

    for value in values:
        internals = torch_axis_protected_internals(value)
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


def _is_permitted_function(internals: TorchAxisProtectedInternals, func: Callable) -> bool:
    return func in internals.permitted_functions


def _normalize_batch_reduction_dims(dim: object, batch_ndim: int) -> int | tuple[int, ...]:
    if dim is None:
        return tuple(range(batch_ndim))
    if isinstance(dim, int):
        return normalize_axis(dim, batch_ndim)
    if isinstance(dim, (tuple, list, torch.Size)) and all(isinstance(item, int) for item in dim):
        dim_tuple = cast("tuple[int, ...]", tuple(dim))
        return normalize_axes(dim_tuple, batch_ndim)

    msg = "reduction dim must be None, an int, or a tuple/list of ints."
    raise TypeError(msg)


class _TorchFunction(Protocol):
    def __call__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        ...


class _BoundTorchFunction(Protocol):
    def __call__(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        ...


class _BoundTorchFunctionWithInternals(Protocol):
    def __call__(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        internals: TorchAxisProtectedInternals,
    ) -> Any:  # noqa: ANN401
        ...


@switchdispatch
def torch_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of torch functions for protected-axis objects."""
    del func, args, kwargs
    return NotImplemented


def torch_function_override(torch_func: _BoundTorchFunction) -> _TorchFunction:
    """Decorator to convert a bound torch function to ``__torch_function__`` shape."""

    @wraps(torch_func)
    def wrapper(
        func: Callable,
        types: tuple[type[Any], ...],  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        return torch_func(func, args, kwargs)

    return wrapper


@overload
def torch_internals_override(
    torch_param_name: str,
) -> Callable[[_BoundTorchFunctionWithInternals], _TorchFunction]: ...


@overload
def torch_internals_override(
    *,
    torch_param_pos: int,
) -> Callable[[_BoundTorchFunctionWithInternals], _TorchFunction]: ...


def torch_internals_override(
    torch_param_name: str | None = None,
    *,
    torch_param_pos: int | None = None,
) -> Callable[[_BoundTorchFunctionWithInternals], _TorchFunction]:
    """Decorator to convert a function taking a protected-axis argument."""
    if torch_param_name is None and torch_param_pos is None:
        msg = "Either torch_param_name or torch_param_pos must be provided."
        raise ValueError(msg)
    if torch_param_name is not None and torch_param_pos is not None:
        msg = "Only one of torch_param_name or torch_param_pos can be provided."
        raise ValueError(msg)

    def decorator(f: _BoundTorchFunctionWithInternals) -> _TorchFunction:
        @wraps(f)
        def wrapper(
            func: Callable,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:  # noqa: ANN401
            mutable_kwargs = dict(kwargs)
            mutable_args = list(args)

            if torch_param_name is not None and torch_param_name in mutable_kwargs:
                protected_arg = mutable_kwargs[torch_param_name]
            elif torch_param_pos is not None and len(mutable_args) > torch_param_pos:
                protected_arg = mutable_args[torch_param_pos]
            else:
                return NotImplemented

            internals = torch_axis_protected_internals(protected_arg)
            if internals is None:
                return NotImplemented

            return f(func, tuple(mutable_args), mutable_kwargs, internals)

        return torch_function_override(wrapper)

    return decorator


@torch_function.register(torch.clone)
@torch_internals_override(torch_param_pos=0)
def protected_clone_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    del args
    memory_format = kwargs.get("memory_format", torch.preserve_format)
    return _apply_unary(internals, lambda _name, value, _axes: func(value, memory_format=memory_format))


@torch_function.multi_register([torch.mean, torch.sum])
@torch_internals_override(torch_param_pos=0)
def protected_batch_reduction_function(  # noqa: PLR0912
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    if not _is_permitted_function(internals, func):
        return NotImplemented

    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    out = kwargs.get("out")
    out_internals = torch_axis_protected_internals(out)
    if out_internals is not None and out_internals.protected_axes != internals.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if out is not None and out_internals is None and len(internals.protected_axes) != 1:
        msg = "non-protected out is only supported for single-field protected objects."
        raise TypeError(msg)

    mutable_args = list(args)
    mutable_kwargs = dict(kwargs)
    results: dict[str, TorchProtectedValue] = {}

    for name, axes_count in internals.protected_axes.items():
        value = internals.values[name]
        batch_ndim = value_ndim(value) - axes_count
        mapped_dim = _normalize_batch_reduction_dims(dim, batch_ndim)

        field_args = list(mutable_args)
        field_kwargs = dict(mutable_kwargs)

        if len(field_args) == 0:
            msg = "torch reduction call is missing the input argument."
            raise TypeError(msg)

        field_args[0] = value
        if len(field_args) > 1:
            field_args[1] = mapped_dim
        else:
            field_kwargs["dim"] = mapped_dim

        if out is not None:
            if out_internals is not None:
                field_kwargs["out"] = out_internals.values[name]
            else:
                field_kwargs["out"] = out

        result = func(*tuple(field_args), **field_kwargs)

        if out is not None:
            continue

        if axes_count == 0 and not hasattr(result, "ndim"):
            result = torch.as_tensor(result)

        original_shape = value_shape(value)
        result_shape = value_shape(result)
        if protected_shape(result_shape, axes_count) != protected_shape(original_shape, axes_count):
            msg = f"Reduction modified protected trailing axes for field {name!r}."
            raise ValueError(msg)

        results[name] = cast("TorchProtectedValue", result)

    if out is not None:
        return out

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)


@torch_function.register(torch.transpose)
@torch_internals_override(torch_param_pos=0)
def protected_transpose_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    dim0 = args[1] if len(args) > 1 else kwargs.get("dim0")
    dim1 = args[2] if len(args) > 2 else kwargs.get("dim1")

    if not isinstance(dim0, int) or not isinstance(dim1, int):
        return NotImplemented

    def op(_name: str, value: TorchProtectedValue, axes_count: int) -> TorchProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        full_dim0 = normalize_axis(dim0, batch_ndim)
        full_dim1 = normalize_axis(dim1, batch_ndim)
        return func(value, full_dim0, full_dim1)

    return _apply_unary(internals, op)


@torch_function.register(torch.permute)
@torch_internals_override(torch_param_pos=0)
def protected_permute_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    dims = args[1] if len(args) > 1 else kwargs.get("dims")
    if not isinstance(dims, (tuple, list, torch.Size)):
        return NotImplemented
    if not all(isinstance(dim, int) for dim in dims):
        return NotImplemented
    if len(dims) != internals.batch_ndim:
        msg = "permute dims must only refer to batch dimensions."
        raise ValueError(msg)

    batch_dims = tuple(dims)

    def op(_name: str, value: TorchProtectedValue, axes_count: int) -> TorchProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        mapped = normalize_axes(batch_dims, batch_ndim)
        full_dims = (*mapped, *range(batch_ndim, value_ndim(value)))
        return func(value, full_dims)

    return _apply_unary(internals, op)


@torch_function.register(torch.adjoint)
@torch_internals_override(torch_param_pos=0)
def protected_adjoint_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    del func, args, kwargs

    def op(_name: str, value: TorchProtectedValue, axes_count: int) -> TorchProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        if batch_ndim < 2:
            msg = "adjoint requires at least 2 batch dimensions."
            raise ValueError(msg)

        result = torch.transpose(cast("Any", value), batch_ndim - 2, batch_ndim - 1)
        return torch.conj(result) if torch.is_complex(result) else result

    return _apply_unary(internals, op)


@torch_function.register(torch.reshape)
@torch_internals_override(torch_param_pos=0)
def protected_reshape_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    shape = args[1] if len(args) > 1 else kwargs.get("shape")
    if shape is None:
        return NotImplemented

    if isinstance(shape, int):
        batch_target_shape = (shape,)
    elif isinstance(shape, (tuple, list, torch.Size)):
        batch_target_shape = tuple(1 if dim is None else dim for dim in shape)
    else:
        msg = "reshape shape must be an int, tuple, list, or torch.Size."
        raise TypeError(msg)

    def op(_name: str, value: TorchProtectedValue, axes_count: int) -> TorchProtectedValue:
        target_shape = (*batch_target_shape, *protected_shape(value_shape(value), axes_count))
        return func(value, target_shape)

    return _apply_unary(internals, op)


@torch_function.register(torch.unsqueeze)
@torch_internals_override(torch_param_pos=0)
def protected_unsqueeze_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    if not isinstance(dim, int):
        return NotImplemented

    def op(_name: str, value: TorchProtectedValue, axes_count: int) -> TorchProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        full_dim = normalize_axis(dim, batch_ndim, allow_endpoint=True)
        return func(value, full_dim)

    return _apply_unary(internals, op)


@torch_function.register(torch.squeeze)
@torch_internals_override(torch_param_pos=0)
def protected_squeeze_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    dim = args[1] if len(args) > 1 else kwargs.get("dim")

    def op(_name: str, value: TorchProtectedValue, axes_count: int) -> TorchProtectedValue:
        batch_ndim = value_ndim(value) - axes_count

        if dim is None:
            shape = value_shape(value)
            squeeze_dims = tuple(i for i, size in enumerate(shape[:batch_ndim]) if size == 1)
        else:
            if isinstance(dim, int):
                dim_tuple = coerce_axis_tuple(dim)
            elif isinstance(dim, (tuple, list, torch.Size)) and all(isinstance(item, int) for item in dim):
                dim_tuple = tuple(dim)
            else:
                msg = "squeeze dim must be an int or tuple/list of ints."
                raise TypeError(msg)
            squeeze_dims = normalize_axes(dim_tuple, batch_ndim)

        result = value
        for axis in sorted(set(squeeze_dims), reverse=True):
            result = func(cast("Any", result), dim=axis)
        return result

    return _apply_unary(internals, op)


@torch_function.multi_register([torch.movedim, torch.moveaxis])
@torch_internals_override(torch_param_pos=0)
def protected_movedim_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    source = args[1] if len(args) > 1 else kwargs.get("source")
    destination = args[2] if len(args) > 2 else kwargs.get("destination")
    if source is None or destination is None:
        return NotImplemented

    if isinstance(source, int):
        source_tuple = coerce_axis_tuple(source)
        source_was_int = True
    elif isinstance(source, (tuple, list, torch.Size)) and all(isinstance(item, int) for item in source):
        source_tuple = tuple(source)
        source_was_int = False
    else:
        return NotImplemented

    if isinstance(destination, int):
        destination_tuple = coerce_axis_tuple(destination)
        destination_was_int = True
    elif isinstance(destination, (tuple, list, torch.Size)) and all(isinstance(item, int) for item in destination):
        destination_tuple = tuple(destination)
        destination_was_int = False
    else:
        return NotImplemented

    def op(_name: str, value: TorchProtectedValue, axes_count: int) -> TorchProtectedValue:
        batch_ndim = value_ndim(value) - axes_count
        mapped_source = normalize_axes(source_tuple, batch_ndim)
        mapped_destination = normalize_axes(destination_tuple, batch_ndim)
        source_arg: int | tuple[int, ...] = mapped_source[0] if source_was_int else mapped_source
        destination_arg: int | tuple[int, ...] = mapped_destination[0] if destination_was_int else mapped_destination
        return func(value, source=source_arg, destination=destination_arg)

    return _apply_unary(internals, op)


@torch_function.multi_register([torch.cat, torch.concat, torch.concatenate])
@torch_function_override
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

    out_internals = torch_axis_protected_internals(out)
    sequence = _extract_protected_value_sequence_internals(values)
    template = sequence.template if sequence.template is not None else out_internals
    if template is None:
        return NotImplemented

    if out_internals is not None and out_internals.protected_axes != template.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if not isinstance(dim, int):
        return NotImplemented

    results: dict[str, TorchProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "cat with protected out requires at least one protected input."
            raise TypeError(msg)

        field_values = sequence.values_by_field[name]
        batch_ndim = value_ndim(template.values[name]) - axes_count
        mapped_dim = normalize_axis(dim, batch_ndim)

        out_value = out_internals.values[name] if out_internals is not None else None
        result = func(field_values, dim=mapped_dim, out=out_value)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, template.protected_axes)
    return template.create(results)


@torch_function.register(torch.stack)
@torch_function_override
def protected_stack_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    mutable_kwargs = dict(kwargs)
    mutable_args = list(args)

    values = tuple(mutable_args[0]) if len(mutable_args) > 0 else tuple(mutable_kwargs["tensors"])
    dim = mutable_kwargs.get("dim", mutable_args[1] if len(mutable_args) > 1 else 0)
    out = mutable_kwargs.get("out")

    out_internals = torch_axis_protected_internals(out)
    sequence = _extract_protected_value_sequence_internals(values)
    template = sequence.template if sequence.template is not None else out_internals
    if template is None:
        return NotImplemented

    if out_internals is not None and out_internals.protected_axes != template.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)

    if not isinstance(dim, int):
        return NotImplemented

    results: dict[str, TorchProtectedValue] = {}
    for name, axes_count in template.protected_axes.items():
        if not sequence.has_protected:
            msg = "stack with protected out requires at least one protected input."
            raise TypeError(msg)

        field_values = sequence.values_by_field[name]
        batch_ndim = value_ndim(template.values[name]) - axes_count
        mapped_dim = normalize_axis(dim, batch_ndim, allow_endpoint=True)

        out_value = out_internals.values[name] if out_internals is not None else None
        result = func(field_values, dim=mapped_dim, out=out_value)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, template.protected_axes)
    return template.create(results)
