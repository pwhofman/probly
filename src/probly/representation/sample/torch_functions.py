"""Torch function implementations for sample tensors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch, wraps
from typing import TYPE_CHECKING, Any, Protocol, overload

import torch

from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from probly.representation.torch_like import TorchLike


class TorchSampleCreator[D: TorchLike](Protocol):
    """Protocol for creating sample tensors."""

    def __call__(self, tensor: D, sample_dim: int, weights: torch.Tensor | None) -> Any:  # noqa: ANN401
        """Create a sample tensor from a torch tensor and a sample dimension."""


@dataclass(frozen=True, slots=True)
class TorchSampleInternals[D: TorchLike]:
    """Internal information about a sample tensor."""

    create: TorchSampleCreator[D]
    tensor: D
    sample_dim: int
    weights: torch.Tensor | None = None


@singledispatch
def torch_sample_internals(_: object) -> TorchSampleInternals | None:
    """Get internals for a sample tensor."""
    return None


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


class _BoundTorchFunctionWithInternals[D: TorchLike](Protocol):
    def __call__(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        create_sample: TorchSampleCreator[D],
        tensor: TorchLike[D],
        sample_dim: int,
        weights: torch.Tensor | None,
    ) -> Any:  # noqa: ANN401
        ...


@switchdispatch
def torch_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of torch functions for sample tensors."""
    del func, args, kwargs
    return NotImplemented


def torch_function_override(
    torch_func: _BoundTorchFunction,
) -> _TorchFunction:
    """Decorator to convert a bound torch function to a torch function."""

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
    torch_sample_param_name: str,
) -> Callable[[_BoundTorchFunctionWithInternals], _TorchFunction]: ...


@overload
def torch_internals_override(
    *,
    torch_sample_param_pos: int,
) -> Callable[[_BoundTorchFunctionWithInternals], _TorchFunction]: ...


def torch_internals_override(
    torch_sample_param_name: str | None = None, *, torch_sample_param_pos: int | None = None
) -> Callable[[_BoundTorchFunctionWithInternals], _TorchFunction]:
    """Decorator to convert a function taking a sample tensor argument."""
    if torch_sample_param_name is None and torch_sample_param_pos is None:
        msg = "Either torch_sample_param_name or torch_sample_param_pos must be provided."
        raise ValueError(msg)
    if torch_sample_param_name is not None and torch_sample_param_pos is not None:
        msg = "Only one of torch_sample_param_name or torch_sample_param_pos can be provided."
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

            if torch_sample_param_name is not None and torch_sample_param_name in mutable_kwargs:
                sample_arg = mutable_kwargs[torch_sample_param_name]
                in_kwargs = True
                param_name = torch_sample_param_name
            elif torch_sample_param_pos is not None and len(mutable_args) > torch_sample_param_pos:
                sample_arg = mutable_args[torch_sample_param_pos]
                in_kwargs = False
                param_pos = torch_sample_param_pos
            else:
                return NotImplemented

            internals = torch_sample_internals(sample_arg)

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
                internals.sample_dim,
                internals.weights,
            )

        return torch_function_override(wrapper)

    return decorator


def _extract_sample_tensor_sequence_internals(
    tensors: tuple[TorchLike, ...],
) -> tuple[list[TorchLike], list[torch.Tensor | None], bool, TorchSampleCreator | None, int | None, int | None]:
    """Extract internals from a sequence of tensors or sample tensors."""
    cast_tensors: list[TorchLike] = []
    weights: list[torch.Tensor | None] = []
    has_sample_tensors = False
    sample_dims: set[int] = set()
    create_sample: TorchSampleCreator | None = None
    sample_ndim: int | None = None

    for tensor in tensors:
        internals = torch_sample_internals(tensor)

        if internals is None:
            cast_tensors.append(tensor)
            weights.append(None)
            continue

        has_sample_tensors = True
        cast_tensors.append(internals.tensor)
        weights.append(internals.weights)
        sample_dims.add(internals.sample_dim)

        if create_sample is None:
            create_sample = internals.create
            sample_ndim = internals.tensor.ndim

    if len(sample_dims) == 1:
        return cast_tensors, weights, has_sample_tensors, create_sample, next(iter(sample_dims)), sample_ndim

    return cast_tensors, weights, has_sample_tensors, None, None, None


@torch_function.register(torch.transpose)
@torch_internals_override(torch_sample_param_pos=0)
def torch_transpose_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: TorchSampleCreator,
    tensor: TorchLike,
    sample_dim: int,
    weights: torch.Tensor | None,
) -> Any:  # noqa: ANN401
    """Implementation of torch.transpose for sample tensors."""
    if len(args) > 2:
        dim0 = args[1]
        dim1 = args[2]
    else:
        dim0 = kwargs.get("dim0")
        dim1 = kwargs.get("dim1")

    if not isinstance(dim0, int) or not isinstance(dim1, int):
        return NotImplemented

    ndim = tensor.ndim
    dim0 = dim0 if dim0 >= 0 else ndim + dim0
    dim1 = dim1 if dim1 >= 0 else ndim + dim1

    if sample_dim == dim0:
        new_sample_dim = dim1
    elif sample_dim == dim1:
        new_sample_dim = dim0
    else:
        new_sample_dim = sample_dim

    res = func(*args, **kwargs)

    return create_sample(res, sample_dim=new_sample_dim, weights=weights)


@torch_function.register(torch.permute)
@torch_internals_override(torch_sample_param_pos=0)
def torch_permute_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: TorchSampleCreator,
    tensor: TorchLike,
    sample_dim: int,
    weights: torch.Tensor | None,
) -> Any:  # noqa: ANN401
    """Implementation of torch.permute for sample tensors."""
    dims = args[1] if len(args) > 1 else kwargs.get("dims")

    if not isinstance(dims, (tuple, list, torch.Size)):
        return NotImplemented

    normalized_dims = tuple(dim if dim >= 0 else tensor.ndim + dim for dim in dims)

    if any(not isinstance(dim, int) for dim in normalized_dims):
        return NotImplemented

    new_sample_dim = normalized_dims.index(sample_dim)
    res = func(*args, **kwargs)

    return create_sample(res, sample_dim=new_sample_dim, weights=weights)


@torch_function.register(torch.adjoint)
@torch_internals_override(torch_sample_param_pos=0)
def torch_adjoint_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    create_sample: TorchSampleCreator,
    tensor: TorchLike,
    sample_dim: int,
    weights: torch.Tensor | None,
) -> Any:  # noqa: ANN401
    """Implementation of torch.adjoint for sample tensors."""
    a_ndim = tensor.ndim

    if sample_dim == a_ndim - 1:
        new_sample_dim = a_ndim - 2
    elif sample_dim == a_ndim - 2:
        new_sample_dim = a_ndim - 1
    else:
        new_sample_dim = sample_dim

    res = func(*args, **kwargs)

    return create_sample(res, sample_dim=new_sample_dim, weights=weights)


@torch_function.multi_register([torch.cat, torch.concat, torch.concatenate])
@torch_function_override
def torch_cat_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of torch.cat and aliases for sample tensors."""
    mutable_kwargs = dict(kwargs)
    mutable_args = list(args)

    tensors = tuple(mutable_args[0]) if len(mutable_args) > 0 else tuple(mutable_kwargs["tensors"])

    dim = mutable_kwargs.get("dim", mutable_args[1] if len(mutable_args) > 1 else 0)
    if not isinstance(dim, int):
        return NotImplemented

    out = mutable_kwargs.get("out")
    out_internals = torch_sample_internals(out)
    cast_tensors, weights, has_sample_tensors, create_sample, sample_dim, sample_ndim = (
        _extract_sample_tensor_sequence_internals(tensors)
    )

    if not has_sample_tensors and out_internals is None:
        return NotImplemented

    if len(mutable_args) > 0:
        mutable_args[0] = cast_tensors
    else:
        mutable_kwargs["tensors"] = cast_tensors

    if out_internals is not None:
        mutable_kwargs["out"] = out_internals.tensor

    res = func(*tuple(mutable_args), **mutable_kwargs)

    if out is not None:
        return out

    if create_sample is None or sample_dim is None:
        return res

    if sample_ndim is not None:
        dim = dim if dim >= 0 else sample_ndim + dim

    if any(w is not None for w in weights):
        if dim != sample_dim:
            msg = "Weighted samples only support cat along the sample dimension."
            raise ValueError(msg)

        weights = torch.concatenate(
            [w if w is not None else torch.ones(cast_tensors[i].shape[sample_dim]) for i, w in enumerate(weights)]
        )
    else:
        weights = None

    return create_sample(res, sample_dim=sample_dim, weights=weights)


@torch_function.register(torch.stack)
@torch_function_override
def torch_stack_function(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of torch.stack for sample tensors."""
    mutable_kwargs = dict(kwargs)
    mutable_args = list(args)

    tensors = tuple(mutable_args[0]) if len(mutable_args) > 0 else tuple(mutable_kwargs["tensors"])
    dim = mutable_kwargs.get("dim", mutable_args[1] if len(mutable_args) > 1 else 0)

    if not isinstance(dim, int):
        return NotImplemented

    out = mutable_kwargs.get("out")
    out_internals = torch_sample_internals(out)
    cast_tensors, weights, has_sample_tensors, create_sample, sample_dim, sample_ndim = (
        _extract_sample_tensor_sequence_internals(tensors)
    )

    if not has_sample_tensors and out_internals is None:
        return NotImplemented

    if len(mutable_args) > 0:
        mutable_args[0] = cast_tensors
    else:
        mutable_kwargs["tensors"] = cast_tensors

    if out_internals is not None:
        mutable_kwargs["out"] = out_internals.tensor

    res = func(*tuple(mutable_args), **mutable_kwargs)

    if out is not None:
        return out

    if create_sample is None or sample_dim is None:
        return res

    if any(w is not None for w in weights):
        msg = "Weighted samples do not support stack."
        raise ValueError(msg)

    input_ndim = sample_ndim
    if input_ndim is None:
        return res

    axis = dim if dim >= 0 else input_ndim + dim + 1
    new_sample_dim = sample_dim + 1 if axis <= sample_dim else sample_dim

    return create_sample(res, sample_dim=new_sample_dim, weights=None)
