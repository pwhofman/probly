"""Torch implementations of spectral uncertainty measures."""

from __future__ import annotations

import torch

from probly.representation.embedding.torch import TorchEmbedding, TorchEmbeddingSample, TorchEmbeddingSampleSample

from ._common import conditional_spectral_entropy, spectral_entropy

type EmbeddingInput = TorchEmbedding | TorchEmbeddingSample | TorchEmbeddingSampleSample | torch.Tensor


def _embedding_tensor(embeddings: EmbeddingInput) -> torch.Tensor:
    if isinstance(embeddings, TorchEmbedding):
        return embeddings.embeddings
    if not isinstance(embeddings, torch.Tensor):
        msg = "embeddings must be a TorchEmbedding or torch.Tensor."
        raise TypeError(msg)
    return embeddings


def _normalize_axes(axes: int | tuple[int, ...], ndim: int, *, name: str = "sample_dim") -> tuple[int, ...]:
    axis_tuple = (axes,) if isinstance(axes, int) else tuple(axes)
    normalized: list[int] = []
    for axis in axis_tuple:
        current = axis + ndim if axis < 0 else axis
        if current < 0 or current >= ndim:
            msg = f"{name} {axis} out of bounds for embeddings with ndim {ndim}."
            raise ValueError(msg)
        normalized.append(current)
    if len(set(normalized)) != len(normalized):
        msg = f"{name} axes must be unique."
        raise ValueError(msg)
    return tuple(normalized)


def _sample_matrix(embeddings: EmbeddingInput, sample_dim: int | tuple[int, ...]) -> torch.Tensor:
    tensor = _embedding_tensor(embeddings)
    if tensor.ndim < 2:
        msg = "embeddings must have at least one batch axis and one embedding axis."
        raise ValueError(msg)
    if not torch.is_floating_point(tensor):
        msg = "embeddings must be floating point."
        raise TypeError(msg)

    batch_ndim = tensor.ndim - 1
    sample_axes = _normalize_axes(sample_dim, batch_ndim)
    batch_axes = tuple(axis for axis in range(batch_ndim) if axis not in sample_axes)
    permuted = tensor.permute((*batch_axes, *sample_axes, batch_ndim))
    batch_shape = permuted.shape[: len(batch_axes)]
    sample_size = 1
    for axis in range(len(batch_axes), len(batch_axes) + len(sample_axes)):
        sample_size *= permuted.shape[axis]
    return permuted.reshape((*batch_shape, sample_size, permuted.shape[-1]))


def _nested_sample_axes(embeddings: TorchEmbeddingSampleSample) -> tuple[TorchEmbedding | torch.Tensor, int, int]:
    """Unpack a nested sample into its embeddings, the sample axis within a group and the group axis."""
    if embeddings.weights is not None or embeddings.tensor.weights is not None:
        msg = "Weighted spectral entropy is not supported."
        raise ValueError(msg)
    inner = embeddings.tensor
    return inner.tensor, inner.sample_dim, embeddings.sample_dim


def _group_axis_after_sample_reduction(group_axis: int, sample_axis: int, ndim: int) -> int:
    """Position of the group axis once the sample axis has been reduced away."""
    (group,) = _normalize_axes(group_axis, ndim, name="group axis")
    sample_axes = _normalize_axes(sample_axis, ndim)
    if group in sample_axes:
        msg = "The group axis must not be one of the sample axes."
        raise ValueError(msg)
    return sum(1 for axis in range(group) if axis not in sample_axes)


def torch_rbf_kernel(
    embeddings: EmbeddingInput,
    *,
    gamma: float = 1.0,
    sample_dim: int | tuple[int, ...] = -1,
    normalized: bool = True,
) -> torch.Tensor:
    """Compute an RBF kernel matrix over embedding samples.

    Args:
        embeddings: Embeddings with shape ``(*batch_shape, *sample_shape, embedding_dim)``.
        gamma: RBF bandwidth parameter.
        sample_dim: Batch axis or axes containing samples. Multiple axes are flattened.
        normalized: Whether rows are L2-normalized. If true, squared distances are computed from dot products as
            ``2 - 2 * dot``.

    Returns:
        Kernel matrices with shape ``(*batch_shape, n, n)``.
    """
    if gamma <= 0:
        msg = "gamma must be positive."
        raise ValueError(msg)

    matrix = _sample_matrix(embeddings, sample_dim)
    similarity = matrix @ matrix.mT
    if normalized:
        distance_squared = (2.0 - 2.0 * similarity).clamp_min(0.0)
    else:
        squared_norm = torch.sum(matrix * matrix, dim=-1, keepdim=True)
        distance_squared = (squared_norm + squared_norm.mT - 2.0 * similarity).clamp_min(0.0)
    return torch.exp(-gamma * distance_squared)


def torch_von_neumann_entropy(kernel: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Compute Von Neumann entropy from positive semidefinite kernel matrices.

    Args:
        kernel: Square kernel matrices with shape ``(*batch_shape, n, n)``.
        eps: Numerical cutoff for trace and eigenvalues.

    Returns:
        Entropies with shape ``batch_shape``.
    """
    if not isinstance(kernel, torch.Tensor):
        msg = "kernel must be a torch.Tensor."
        raise TypeError(msg)
    if kernel.ndim < 2 or kernel.shape[-1] != kernel.shape[-2]:
        msg = "kernel must have shape (*batch_shape, n, n)."
        raise ValueError(msg)
    if eps < 0:
        msg = "eps must be non-negative."
        raise ValueError(msg)

    batch_shape = kernel.shape[:-2]
    if kernel.shape[-1] <= 1:
        return torch.zeros(batch_shape, dtype=kernel.dtype, device=kernel.device)

    trace = torch.diagonal(kernel, dim1=-2, dim2=-1).sum(dim=-1)
    valid_trace = trace > eps
    density = kernel / trace.clamp_min(eps)[..., None, None]
    eigenvalues = torch.linalg.eigvalsh(density).clamp_min(0.0)
    terms = torch.where(eigenvalues > eps, torch.special.entr(eigenvalues), torch.zeros_like(eigenvalues))
    entropy = terms.sum(dim=-1)
    return torch.where(valid_trace, entropy, torch.zeros_like(entropy))


@spectral_entropy.register(TorchEmbedding | TorchEmbeddingSample | TorchEmbeddingSampleSample)
def torch_spectral_entropy(
    embeddings: EmbeddingInput,
    *,
    sample_dim: int | tuple[int, ...] = -1,
    gamma: float = 1.0,
    normalized: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute spectral uncertainty as Von Neumann entropy of an RBF kernel.

    Args:
        embeddings: Embeddings with one protected trailing embedding axis. Samples carry their own sample axes; for
            a sample of embedding samples, the groups and the samples within them are flattened into one set.
        sample_dim: Batch axis or axes containing samples. Multiple axes are flattened.
        gamma: RBF bandwidth parameter.
        normalized: Whether embeddings are L2-normalized.
        eps: Numerical cutoff for trace and eigenvalues.

    Returns:
        Spectral entropy over the remaining batch shape.
    """
    if isinstance(embeddings, TorchEmbeddingSampleSample):
        embeddings, sample_axis, group_axis = _nested_sample_axes(embeddings)
        sample_dim = (sample_axis, group_axis)

    if isinstance(embeddings, TorchEmbeddingSample):
        sample_dim = embeddings.sample_dim
        embeddings = embeddings.tensor

    if isinstance(embeddings, TorchEmbedding):
        embeddings = embeddings.embeddings

    kernel = torch_rbf_kernel(embeddings, gamma=gamma, sample_dim=sample_dim, normalized=normalized)
    return torch_von_neumann_entropy(kernel, eps=eps)


@conditional_spectral_entropy.register(TorchEmbeddingSampleSample)
def torch_conditional_spectral_entropy(
    embeddings: TorchEmbeddingSampleSample,
    *,
    gamma: float = 1.0,
    normalized: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the mean spectral entropy within the groups of a nested torch sample."""
    inner, sample_axis, group_axis = _nested_sample_axes(embeddings)
    ndim = _embedding_tensor(inner).ndim - 1
    group_axis_after_reduction = _group_axis_after_sample_reduction(group_axis, sample_axis, ndim)
    group_entropies = torch_spectral_entropy(inner, sample_dim=sample_axis, gamma=gamma, normalized=normalized, eps=eps)
    return torch.mean(group_entropies, dim=group_axis_after_reduction)
