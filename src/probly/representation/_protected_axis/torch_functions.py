"""Torch function implementations for protected-axis tensors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

import torch

from probly.representation._protected_axis._common_functions import coerce_axis_tuple, normalize_axes, normalize_axis
from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable


class TorchAxisProtectedCreator(Protocol):
    """Protocol for creating protected-axis representations."""

    def __call__(self, tensor: torch.Tensor) -> Any:  # noqa: ANN401
        """Create an object from a protected tensor."""


@runtime_checkable
class _SupportsProtectedInternals(Protocol):
    protected_axes: int

    def protected_tensor(self) -> torch.Tensor:
        """Return the tensor with protected trailing axes."""

    def with_protected_tensor(self, tensor: torch.Tensor) -> Any:  # noqa: ANN401
        """Create a new object with a replaced protected tensor."""


@dataclass(frozen=True, slots=True)
class TorchAxisProtectedInternals:
    """Internal information about a protected-axis representation."""

    create: TorchAxisProtectedCreator
    tensor: torch.Tensor
    protected_axes: int

    @property
    def batch_ndim(self) -> int:
        return self.tensor.ndim - self.protected_axes

    @property
    def protected_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape[-self.protected_axes :])


def torch_axis_protected_internals(obj: object) -> TorchAxisProtectedInternals | None:
    """Get internals for protected-axis representations."""
    if not isinstance(obj, _SupportsProtectedInternals):
        return None

    tensor = obj.protected_tensor()
    protected_axes = obj.protected_axes

    if not isinstance(tensor, torch.Tensor) or not isinstance(protected_axes, int):
        return None

    create = obj.with_protected_tensor
    return TorchAxisProtectedInternals(create=create, tensor=tensor, protected_axes=protected_axes)


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
        create_protected: TorchAxisProtectedCreator,
        tensor: torch.Tensor,
        protected_axes: int,
    ) -> Any:  # noqa: ANN401
        ...


@switchdispatch
def torch_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of torch functions for protected-axis tensors."""
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
            in_kwargs: bool
            param_name: str
            param_pos: int

            if torch_param_name is not None and torch_param_name in mutable_kwargs:
                protected_arg = mutable_kwargs[torch_param_name]
                in_kwargs = True
                param_name = torch_param_name
            elif torch_param_pos is not None and len(mutable_args) > torch_param_pos:
                protected_arg = mutable_args[torch_param_pos]
                in_kwargs = False
                param_pos = torch_param_pos
            else:
                return NotImplemented

            internals = torch_axis_protected_internals(protected_arg)

            if internals is None:
                return NotImplemented

            if in_kwargs:
                mutable_kwargs[param_name] = internals.tensor
            else:
                mutable_args[param_pos] = internals.tensor

            return f(
                func,
                tuple(mutable_args),
                mutable_kwargs,
                internals.create,
                internals.tensor,
                internals.protected_axes,
            )

        return torch_function_override(wrapper)

    return decorator


@torch_function.register(torch.clone)
@torch_internals_override(torch_param_pos=0)
def protected_clone_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_protected: TorchAxisProtectedCreator,
    tensor: torch.Tensor,
    protected_axes: int,  # noqa: ARG001
) -> Any:  # noqa: ANN401
    """Implementation of ``torch.clone`` for protected-axis tensors."""
    del args
    memory_format = kwargs.get("memory_format", torch.preserve_format)
    result = func(tensor, memory_format=memory_format)
    return create_protected(result)


@torch_function.register(torch.transpose)
@torch_internals_override(torch_param_pos=0)
def protected_transpose_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_protected: TorchAxisProtectedCreator,
    tensor: torch.Tensor,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of ``torch.transpose`` for protected-axis tensors."""
    dim0 = args[1] if len(args) > 1 else kwargs.get("dim0")
    dim1 = args[2] if len(args) > 2 else kwargs.get("dim1")

    if not isinstance(dim0, int) or not isinstance(dim1, int):
        return NotImplemented

    batch_ndim = tensor.ndim - protected_axes
    full_dim0 = normalize_axis(dim0, batch_ndim)
    full_dim1 = normalize_axis(dim1, batch_ndim)
    result = func(tensor, full_dim0, full_dim1)
    return create_protected(result)


@torch_function.register(torch.permute)
@torch_internals_override(torch_param_pos=0)
def protected_permute_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_protected: TorchAxisProtectedCreator,
    tensor: torch.Tensor,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of ``torch.permute`` for protected-axis tensors."""
    dims = args[1] if len(args) > 1 else kwargs.get("dims")

    if not isinstance(dims, (tuple, list, torch.Size)):
        return NotImplemented

    if not all(isinstance(dim, int) for dim in dims):
        return NotImplemented

    batch_ndim = tensor.ndim - protected_axes
    batch_dims = normalize_axes(tuple(dims), batch_ndim)
    if len(batch_dims) != batch_ndim:
        msg = "permute dims must only refer to batch dimensions."
        raise ValueError(msg)

    full_dims = (*batch_dims, *range(batch_ndim, tensor.ndim))
    result = func(tensor, full_dims)
    return create_protected(result)


@torch_function.register(torch.adjoint)
@torch_internals_override(torch_param_pos=0)
def protected_adjoint_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_protected: TorchAxisProtectedCreator,
    tensor: torch.Tensor,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of ``torch.adjoint`` for protected-axis tensors."""
    del func, args, kwargs
    batch_ndim = tensor.ndim - protected_axes
    if batch_ndim < 2:
        msg = "adjoint requires at least 2 batch dimensions."
        raise ValueError(msg)

    result = torch.transpose(tensor, batch_ndim - 2, batch_ndim - 1)
    if torch.is_complex(result):
        result = torch.conj(result)
    return create_protected(result)


@torch_function.register(torch.reshape)
@torch_internals_override(torch_param_pos=0)
def protected_reshape_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_protected: TorchAxisProtectedCreator,
    tensor: torch.Tensor,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of ``torch.reshape`` for protected-axis tensors."""
    shape = args[1] if len(args) > 1 else kwargs.get("shape")
    if shape is None:
        return NotImplemented

    if isinstance(shape, int):
        batch_shape = (shape,)
    elif isinstance(shape, (tuple, list, torch.Size)):
        batch_shape = tuple(1 if dim is None else dim for dim in shape)
    else:
        msg = "reshape shape must be an int, tuple, list, or torch.Size."
        raise TypeError(msg)

    protected_shape = tensor.shape[-protected_axes:]
    full_shape = (*batch_shape, *protected_shape)
    result = func(tensor, full_shape)
    return create_protected(result)


@torch_function.register(torch.unsqueeze)
@torch_internals_override(torch_param_pos=0)
def protected_unsqueeze_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_protected: TorchAxisProtectedCreator,
    tensor: torch.Tensor,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of ``torch.unsqueeze`` for protected-axis tensors."""
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    if not isinstance(dim, int):
        return NotImplemented

    batch_ndim = tensor.ndim - protected_axes
    full_dim = normalize_axis(dim, batch_ndim, allow_endpoint=True)
    result = func(tensor, full_dim)
    return create_protected(result)


@torch_function.register(torch.squeeze)
@torch_internals_override(torch_param_pos=0)
def protected_squeeze_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_protected: TorchAxisProtectedCreator,
    tensor: torch.Tensor,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of ``torch.squeeze`` for protected-axis tensors."""
    del func
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    batch_ndim = tensor.ndim - protected_axes

    if dim is None:
        squeeze_dims = tuple(i for i, size in enumerate(tensor.shape[:batch_ndim]) if size == 1)
    else:
        if isinstance(dim, int):
            dim_tuple = coerce_axis_tuple(dim)
        elif isinstance(dim, (tuple, list, torch.Size)) and all(isinstance(d, int) for d in dim):
            dim_tuple = tuple(dim)
        else:
            return NotImplemented

        squeeze_dims = normalize_axes(dim_tuple, batch_ndim)

    result = tensor
    for axis in sorted(set(squeeze_dims), reverse=True):
        result = torch.squeeze(result, dim=axis)

    return create_protected(result)


@torch_function.multi_register([torch.movedim, torch.moveaxis])
@torch_internals_override(torch_param_pos=0)
def protected_movedim_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_protected: TorchAxisProtectedCreator,
    tensor: torch.Tensor,
    protected_axes: int,
) -> Any:  # noqa: ANN401
    """Implementation of ``torch.movedim`` / ``torch.moveaxis`` for protected-axis tensors."""
    source = args[1] if len(args) > 1 else kwargs.get("source")
    destination = args[2] if len(args) > 2 else kwargs.get("destination")

    if source is None or destination is None:
        return NotImplemented

    if isinstance(source, int):
        source_tuple = coerce_axis_tuple(source)
        source_was_int = True
    elif isinstance(source, (tuple, list, torch.Size)) and all(isinstance(s, int) for s in source):
        source_tuple = tuple(source)
        source_was_int = False
    else:
        return NotImplemented

    if isinstance(destination, int):
        destination_tuple = coerce_axis_tuple(destination)
        destination_was_int = True
    elif isinstance(destination, (tuple, list, torch.Size)) and all(isinstance(d, int) for d in destination):
        destination_tuple = tuple(destination)
        destination_was_int = False
    else:
        return NotImplemented

    batch_ndim = tensor.ndim - protected_axes
    mapped_source = normalize_axes(source_tuple, batch_ndim)
    mapped_destination = normalize_axes(destination_tuple, batch_ndim)

    source_arg: int | tuple[int, ...] = mapped_source[0] if source_was_int else mapped_source
    destination_arg: int | tuple[int, ...] = mapped_destination[0] if destination_was_int else mapped_destination
    result = func(tensor, source=source_arg, destination=destination_arg)
    return create_protected(result)
