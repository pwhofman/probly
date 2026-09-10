"""Torch-function implementations for protected-axis values."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

import numpy as np
import torch

from probly.representation._protected_axis._common_functions import (
    AxisProtectedCreator,
    AxisProtectedInternals,
    FunctionOverride as _TorchFunction,
    ProtectedValueSequenceInternals,
    SupportsProtectedInternals,
    apply_structural_op as _apply_structural_op,
    apply_unary as _apply_unary,
    coerce_axis_tuple,
    extract_axis_protected_internals,
    extract_protected_value_sequence_internals,
    has_numpy_protected_value as _has_numpy_protected_value,
    map_batch_axes,
    normalize_axes,
    normalize_axis,
    normalize_batch_reduction_axes,
    protected_shape,
    validate_batch_sync as _validate_batch_sync,
    value_ndim,
    value_shape,
)
from probly.representation.torch_functions import torch_average
from probly.representation.torch_like import TorchLike
from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable


type TorchProtectedValue = TorchLike[Any] | torch.Tensor | np.ndarray


type TorchAxisProtectedCreator = AxisProtectedCreator[TorchProtectedValue, Any]
type TorchAxisProtectedInternals = AxisProtectedInternals[TorchProtectedValue, Any]


def torch_axis_protected_internals(
    obj: object, func: Callable | None = None, *, check_is_permitted: bool = False
) -> TorchAxisProtectedInternals | None:
    """Extract protected-axis internals from object."""
    if not isinstance(obj, SupportsProtectedInternals):
        return None
    # Dispatch establishes the value family; runtime protocol checks only test members.
    return extract_axis_protected_internals(
        cast("SupportsProtectedInternals[TorchProtectedValue, Any]", obj),
        func,
        check_is_permitted=check_is_permitted,
    )


def _extract_protected_value_sequence_internals(
    values: tuple[object, ...], func: Callable | None = None
) -> ProtectedValueSequenceInternals[TorchProtectedValue, Any]:
    """Extract and align protected values for sequence operations."""
    return extract_protected_value_sequence_internals(values, func, extract=torch_axis_protected_internals)


def _normalize_batch_reduction_dims(dim: object, batch_ndim: int) -> int | tuple[int, ...]:
    return normalize_batch_reduction_axes(dim, batch_ndim, parameter_name="dim")


def _expand_average_weights_for_protected_axes(
    weights: object,
    value: TorchProtectedValue,
    axes_count: int,
) -> object:
    if not isinstance(weights, torch.Tensor) or axes_count == 0:
        return weights

    batch_ndim = value_ndim(value) - axes_count
    if weights.ndim == batch_ndim:
        return weights.reshape((*weights.shape, *((1,) * axes_count)))

    return weights


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
    *,
    check_is_permitted: bool = False,
) -> Callable[[_BoundTorchFunctionWithInternals], _TorchFunction]: ...


@overload
def torch_internals_override(
    *,
    torch_param_pos: int,
    check_is_permitted: bool = False,
) -> Callable[[_BoundTorchFunctionWithInternals], _TorchFunction]: ...


def torch_internals_override(
    torch_param_name: str | None = None,
    *,
    torch_param_pos: int | None = None,
    check_is_permitted: bool = False,
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

            internals = torch_axis_protected_internals(
                protected_arg,
                func,
                check_is_permitted=check_is_permitted,
            )
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
    if _has_numpy_protected_value(internals):
        return NotImplemented

    memory_format = kwargs.get("memory_format", torch.preserve_format)
    return _apply_unary(internals, lambda _name, value, _axes: func(value, memory_format=memory_format))


@torch_function.multi_register([torch.mean, torch.sum, torch_average])
@torch_internals_override(torch_param_pos=0, check_is_permitted=True)
def protected_batch_reduction_function(  # noqa: PLR0912
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    if _has_numpy_protected_value(internals):
        return NotImplemented

    dim = args[1] if len(args) > 1 else kwargs.get("dim", kwargs.get("axis"))
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
        elif "axis" in field_kwargs and "dim" not in field_kwargs:
            field_kwargs["axis"] = mapped_dim
        else:
            field_kwargs["dim"] = mapped_dim

        if out is not None:
            if out_internals is not None:
                field_kwargs["out"] = out_internals.values[name]
            else:
                field_kwargs["out"] = out

        if func is torch_average and "weights" in field_kwargs:
            field_kwargs["weights"] = _expand_average_weights_for_protected_axes(
                field_kwargs["weights"],
                value,
                axes_count,
            )

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
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, full_dim0, full_dim1),
            lambda field_value: np.swapaxes(field_value, full_dim0, full_dim1),
        )

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
        full_dims = map_batch_axes(value, axes_count, batch_dims)
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, full_dims),
            lambda field_value: np.transpose(field_value, axes=full_dims),
        )

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

        if isinstance(value, np.ndarray):
            return np.swapaxes(value, batch_ndim - 2, batch_ndim - 1)

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
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, target_shape),
            lambda field_value: np.reshape(field_value, target_shape),
        )

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
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, full_dim),
            lambda field_value: np.expand_dims(field_value, axis=full_dim),
        )

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
        shape = value_shape(value)

        if dim is None:
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

        squeeze_dims = tuple(sorted(set(squeeze_dims)))
        if isinstance(value, np.ndarray):
            numpy_squeeze_dims = tuple(axis for axis in squeeze_dims if shape[axis] == 1)
            return np.squeeze(value, axis=numpy_squeeze_dims) if numpy_squeeze_dims else value

        result = value
        for axis in reversed(squeeze_dims):
            result = func(result, dim=axis)
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
        return _apply_structural_op(
            value,
            lambda field_value: func(field_value, source=source_arg, destination=destination_arg),
            lambda field_value: np.moveaxis(field_value, source_arg, destination_arg),
        )

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
        if isinstance(template.values[name], np.ndarray):
            result = np.concatenate(
                cast("Any", field_values),
                axis=mapped_dim,
                out=cast("Any", out_value),
            )
        else:
            result = func(field_values, dim=mapped_dim, out=out_value)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, template.protected_axes)
    return template.create(results)


@torch_function.register(torch.gather)
@torch_internals_override(torch_param_pos=0)
def protected_gather_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    internals: TorchAxisProtectedInternals,
) -> Any:  # noqa: ANN401
    if _has_numpy_protected_value(internals):
        return NotImplemented

    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    index = args[2] if len(args) > 2 else kwargs.get("index")
    out = kwargs.get("out")
    sparse_grad = kwargs.get("sparse_grad", False)

    if not isinstance(dim, int) or not isinstance(index, torch.Tensor):
        return NotImplemented
    if index.ndim != internals.batch_ndim:
        msg = "gather index must have the same ndim as the protected object's batch dimensions."
        raise ValueError(msg)

    out_internals = torch_axis_protected_internals(out)
    if out_internals is not None and out_internals.protected_axes != internals.protected_axes:
        msg = "out must use the same protected_axes layout as input values."
        raise ValueError(msg)
    if out is not None and out_internals is None and len(internals.protected_axes) != 1:
        msg = "non-protected out is only supported for single-field protected objects."
        raise TypeError(msg)

    results: dict[str, TorchProtectedValue] = {}
    for name, axes_count in internals.protected_axes.items():
        value = internals.values[name]
        batch_ndim = value_ndim(value) - axes_count
        mapped_dim = normalize_axis(dim, batch_ndim)

        field_index = index
        for _ in range(axes_count):
            field_index = field_index.unsqueeze(-1)
        if axes_count > 0:
            target_shape = (*index.shape, *protected_shape(value_shape(value), axes_count))
            field_index = field_index.expand(target_shape)

        out_value = out_internals.values[name] if out_internals is not None else out
        result = func(value, mapped_dim, field_index, sparse_grad=sparse_grad, out=out_value)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, internals.protected_axes)
    return internals.create(results)


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
        if isinstance(template.values[name], np.ndarray):
            result = np.stack(
                cast("Any", field_values),
                axis=mapped_dim,
                out=cast("Any", out_value),
            )
        else:
            result = func(field_values, dim=mapped_dim, out=out_value)
        if out_value is None:
            results[name] = result

    if out is not None:
        return out

    _validate_batch_sync(results, template.protected_axes)
    return template.create(results)
