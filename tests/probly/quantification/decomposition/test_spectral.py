"""Tests for spectral uncertainty decomposition."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification import conditional_spectral_entropy, decompose, spectral_entropy
from probly.quantification.decomposition import SpectralDecomposition
from probly.representation.embedding.torch import TorchEmbedding, TorchEmbeddingSample, TorchEmbeddingSampleSample


def _nested_sample(tensor: torch.Tensor, *, group_dim: int, sample_dim: int) -> TorchEmbeddingSampleSample:
    inner = TorchEmbeddingSample(tensor=TorchEmbedding(tensor), sample_dim=sample_dim)
    return TorchEmbeddingSampleSample(tensor=inner, sample_dim=group_dim)


def test_spectral_decomposition_returns_additive_components() -> None:
    embeddings = _nested_sample(
        torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
            ]
        ),
        group_dim=1,
        sample_dim=2,
    )

    decomposition = decompose(embeddings)

    assert isinstance(decomposition, SpectralDecomposition)
    assert decomposition.total.shape == (2,)
    assert decomposition.aleatoric.shape == (2,)
    assert torch.allclose(decomposition.epistemic, decomposition.total - decomposition.aleatoric)


def test_spectral_decomposition_matches_measures() -> None:
    embeddings = _nested_sample(torch.randn(2, 3, 4, 5), group_dim=1, sample_dim=2)

    decomposition = decompose(embeddings, gamma=0.5)

    assert torch.allclose(decomposition.total, spectral_entropy(embeddings, gamma=0.5))
    assert torch.allclose(decomposition.aleatoric, conditional_spectral_entropy(embeddings, gamma=0.5))


def test_decompose_registered_for_nested_embedding_samples() -> None:
    embeddings = _nested_sample(torch.randn(2, 3, 4, 5), group_dim=1, sample_dim=2)

    decomposition = decompose(embeddings)

    assert decomposition.total.shape == (2,)
    assert decomposition.aleatoric.shape == (2,)


def test_spectral_decomposition_rejects_weighted_samples() -> None:
    embeddings = TorchEmbedding(torch.randn(2, 3, 4))
    inner = TorchEmbeddingSample(tensor=embeddings, sample_dim=1, weights=torch.ones(3))
    outer = TorchEmbeddingSampleSample(tensor=inner, sample_dim=0)

    with pytest.raises(ValueError, match="Weighted"):
        _ = decompose(outer).total
