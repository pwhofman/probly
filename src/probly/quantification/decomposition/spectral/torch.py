"""Torch spectral uncertainty decompositions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import torch

from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import AdditiveDecomposition
from probly.quantification.measure.spectral import spectral_entropy
from probly.representation.embedding import TorchEmbedding, TorchEmbeddingSampleSample


def _embedding_ndim(embeddings: TorchEmbedding | torch.Tensor) -> int:
    if isinstance(embeddings, TorchEmbedding):
        return embeddings.ndim
    if isinstance(embeddings, torch.Tensor):
        if embeddings.ndim < 1:
            msg = "embeddings must include a trailing embedding axis."
            raise ValueError(msg)
        return embeddings.ndim - 1
    msg = "embeddings must be a TorchEmbedding or torch.Tensor."
    raise TypeError(msg)


def _normalize_axis(axis: int, ndim: int, *, name: str) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        msg = f"{name} {axis} out of bounds for embeddings with ndim {ndim}."
        raise ValueError(msg)
    return normalized


def _group_axis_after_sample_reduction(group_dim: int, sample_dim: int | tuple[int, ...], ndim: int) -> int:
    group_axis = _normalize_axis(group_dim, ndim, name="group_dim")
    raw_sample_axes = (sample_dim,) if isinstance(sample_dim, int) else sample_dim
    sample_axes = tuple(_normalize_axis(axis, ndim, name="sample_dim") for axis in raw_sample_axes)
    if group_axis in sample_axes:
        msg = "group_dim must not be included in sample_dim."
        raise ValueError(msg)
    if len(set(sample_axes)) != len(sample_axes):
        msg = "sample_dim axes must be unique."
        raise ValueError(msg)
    return sum(1 for axis in range(group_axis) if axis not in sample_axes)


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class SpectralDecomposition(AdditiveDecomposition[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Spectral decomposition into total, aleatoric, and epistemic uncertainty."""

    total_uncertainty: torch.Tensor
    aleatoric_uncertainty: torch.Tensor

    @override
    @property
    def _total(self) -> torch.Tensor:
        """The total spectral uncertainty."""
        return self.total_uncertainty

    @override
    @property
    def _aleatoric(self) -> torch.Tensor:
        """The aleatoric spectral uncertainty."""
        return self.aleatoric_uncertainty


def spectral_decomposition(
    embeddings: TorchEmbedding | torch.Tensor,
    *,
    group_dim: int,
    sample_dim: int | tuple[int, ...],
    gamma: float = 1.0,
    normalized: bool = True,
    eps: float = 1e-12,
) -> SpectralDecomposition:
    """Decompose spectral uncertainty over grouped embedding samples.

    Args:
        embeddings: Embeddings with a trailing protected embedding axis.
        group_dim: Batch axis containing uncertainty groups, such as clarification samples.
        sample_dim: Batch axis or axes containing samples within each group.
        gamma: RBF bandwidth parameter.
        normalized: Whether embeddings are L2-normalized.
        eps: Numerical cutoff for trace and eigenvalues.

    Returns:
        Additive spectral uncertainty decomposition.
    """
    ndim = _embedding_ndim(embeddings)
    group_axis = _normalize_axis(group_dim, ndim, name="group_dim")
    group_axis_after_reduction = _group_axis_after_sample_reduction(group_dim, sample_dim, ndim)
    sample_axes = (sample_dim,) if isinstance(sample_dim, int) else tuple(sample_dim)

    total = spectral_entropy(
        embeddings,
        sample_dim=(*sample_axes, group_axis),
        gamma=gamma,
        normalized=normalized,
        eps=eps,
    )
    group_entropies = spectral_entropy(
        embeddings,
        sample_dim=sample_dim,
        gamma=gamma,
        normalized=normalized,
        eps=eps,
    )
    aleatoric = torch.mean(group_entropies, dim=group_axis_after_reduction)
    return SpectralDecomposition(total_uncertainty=total, aleatoric_uncertainty=aleatoric)


@decompose.register(TorchEmbeddingSampleSample)
def torch_embedding_sample_sample_spectral_decomposition(
    embeddings: TorchEmbeddingSampleSample,
    *,
    gamma: float = 1.0,
    normalized: bool = True,
    eps: float = 1e-12,
) -> SpectralDecomposition:
    """Decompose nested embedding samples via spectral uncertainty."""
    if embeddings.weights is not None or embeddings.tensor.weights is not None:
        msg = "Weighted spectral decomposition is not supported."
        raise ValueError(msg)
    return spectral_decomposition(
        embeddings.tensor.tensor,
        group_dim=embeddings.sample_dim,
        sample_dim=embeddings.tensor.sample_dim,
        gamma=gamma,
        normalized=normalized,
        eps=eps,
    )
