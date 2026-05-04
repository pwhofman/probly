"""Torch implementations of spectral uncertainty measures."""

from __future__ import annotations

import torch

from probly.quantification._quantification import measure
from probly.representation.embedding import TorchEmbedding
from probly.representation.embedding.torch import TorchEmbeddingSample

type EmbeddingInput = TorchEmbedding | torch.Tensor


def _embedding_tensor(embeddings: EmbeddingInput) -> torch.Tensor:
    if isinstance(embeddings, TorchEmbedding):
        return embeddings.embeddings
    if not isinstance(embeddings, torch.Tensor):
        msg = "embeddings must be a TorchEmbedding or torch.Tensor."
        raise TypeError(msg)
    return embeddings


def _normalize_axes(axes: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    axis_tuple = (axes,) if isinstance(axes, int) else tuple(axes)
    normalized: list[int] = []
    for axis in axis_tuple:
        current = axis + ndim if axis < 0 else axis
        if current < 0 or current >= ndim:
            msg = f"sample_dim {axis} out of bounds for embeddings with ndim {ndim}."
            raise ValueError(msg)
        normalized.append(current)
    if len(set(normalized)) != len(normalized):
        msg = "sample_dim axes must be unique."
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


def rbf_kernel(
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


def von_neumann_entropy(kernel: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
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
    terms = torch.where(eigenvalues > eps, eigenvalues * torch.log(eigenvalues), torch.zeros_like(eigenvalues))
    entropy = -terms.sum(dim=-1)
    return torch.where(valid_trace, entropy, torch.zeros_like(entropy))


@measure.register(TorchEmbedding | TorchEmbeddingSample)
def spectral_entropy(
    embeddings: EmbeddingInput,
    *,
    sample_dim: int | tuple[int, ...] = -1,
    gamma: float = 1.0,
    normalized: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute spectral uncertainty as Von Neumann entropy of an RBF kernel.

    Args:
        embeddings: Embeddings with one protected trailing embedding axis.
        sample_dim: Batch axis or axes containing samples. Multiple axes are flattened.
        gamma: RBF bandwidth parameter.
        normalized: Whether embeddings are L2-normalized.
        eps: Numerical cutoff for trace and eigenvalues.

    Returns:
        Spectral entropy over the remaining batch shape.
    """
    if isinstance(embeddings, TorchEmbeddingSample):
        sample_dim = embeddings.sample_dim
        embeddings = embeddings.tensor

    if isinstance(embeddings, TorchEmbedding):
        embeddings = embeddings.embeddings

    kernel = rbf_kernel(embeddings, gamma=gamma, sample_dim=sample_dim, normalized=normalized)
    return von_neumann_entropy(kernel, eps=eps)
