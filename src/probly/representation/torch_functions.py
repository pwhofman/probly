"""Additional torch-like functions."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch.overrides import handle_torch_function, has_torch_function


def _normalize_dims(dim: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    if dim is None:
        return tuple(range(ndim))

    dims = dim if isinstance(dim, tuple) else (dim,)
    normalized = tuple(d + ndim if d < 0 else d for d in dims)

    if any(d < 0 or d >= ndim for d in normalized):
        msg = f"dim {dim} out of bounds for tensor with ndim {ndim}."
        raise IndexError(msg)
    if len(set(normalized)) != len(normalized):
        msg = f"duplicate value in dim {dim}."
        raise ValueError(msg)

    return normalized


def _floating_tensor(tensor: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        return tensor

    dtype = torch.result_type(tensor, weights) if weights is not None else torch.get_default_dtype()
    if not torch.is_floating_point(torch.empty((), dtype=dtype)) and not torch.is_complex(torch.empty((), dtype=dtype)):
        dtype = torch.get_default_dtype()
    return tensor.to(dtype=dtype)


def _broadcast_average_weights(
    tensor: torch.Tensor,
    weights: torch.Tensor,
    dims: tuple[int, ...],
) -> torch.Tensor:
    if weights.ndim == 0:
        return weights

    if tuple(weights.shape) == tuple(tensor.shape):
        return weights

    if len(dims) == 1 and tuple(weights.shape) == (tensor.shape[dims[0]],):
        shape = [1] * tensor.ndim
        shape[dims[0]] = weights.shape[0]
        return weights.reshape(shape)

    reduced_shape = tuple(tensor.shape[d] for d in dims)
    if tuple(weights.shape) == reduced_shape:
        shape = [1] * tensor.ndim
        for axis, size in zip(dims, weights.shape, strict=True):
            shape[axis] = size
        return weights.reshape(shape)

    try:
        return torch.broadcast_to(weights, tensor.shape)
    except RuntimeError as error:
        msg = "weights must be broadcastable to the tensor or match the reduced dimensions."
        raise ValueError(msg) from error


def torch_average(
    tensor: torch.Tensor,
    /,
    dim: int | tuple[int, ...] | None = None,
    *,
    axis: int | tuple[int, ...] | None = None,
    weights: torch.Tensor | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a possibly weighted average over tensor dimensions.

    Args:
        tensor: Tensor to average.
        dim: Dimension or dimensions to reduce. If omitted, all dimensions are reduced.
        axis: Alias for ``dim`` for NumPy-style call sites.
        weights: Optional weights. Weights may have the same shape as ``input``, be
            broadcastable to ``input``, or match the reduced dimensions.
        keepdim: Whether reduced dimensions are retained with size one.
        out: Optional output tensor.

    Returns:
        The weighted average.
    """
    if axis is not None:
        if dim is not None:
            msg = "Cannot specify both dim and axis."
            raise TypeError(msg)
        dim = axis
        axis = None

    relevant_args = (tensor, weights, out)
    if has_torch_function(relevant_args):
        return handle_torch_function(
            torch_average,
            relevant_args,
            tensor,
            dim,
            axis=axis,
            weights=weights,
            keepdim=keepdim,
            out=out,
        )

    dims = _normalize_dims(dim, tensor.ndim)
    weight_tensor = None if weights is None else torch.as_tensor(weights, device=tensor.device)
    tensor = _floating_tensor(tensor, weight_tensor)

    if weight_tensor is None:
        return torch.mean(tensor, dim=dims, keepdim=keepdim, out=out)

    weight_tensor = weight_tensor.to(device=tensor.device)
    if not torch.is_floating_point(weight_tensor) and not torch.is_complex(weight_tensor):
        weight_tensor = weight_tensor.to(dtype=tensor.dtype)

    broadcast_weights = _broadcast_average_weights(tensor, weight_tensor, dims)
    weighted_sum = torch.sum(tensor * broadcast_weights, dim=dims, keepdim=keepdim)
    weight_sum = torch.sum(broadcast_weights, dim=dims, keepdim=keepdim)

    if bool(torch.any(weight_sum == 0)):
        msg = "Weights sum to zero, cannot be normalized."
        raise ZeroDivisionError(msg)

    result = weighted_sum / weight_sum
    if out is not None:
        cast("Any", out).copy_(result)
        return out
    return result


__all__ = ["torch_average"]
