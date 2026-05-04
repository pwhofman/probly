"""Tests for spectral uncertainty decomposition."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification import decompose
from probly.quantification.decomposition.spectral.torch import spectral_decomposition
from probly.representation.embedding import TorchEmbedding, TorchEmbeddingSample, TorchEmbeddingSampleSample


def test_spectral_decomposition_returns_additive_components() -> None:
    embeddings = TorchEmbedding(
        torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
            ]
        )
    )

    decomposition = spectral_decomposition(embeddings, group_dim=1, sample_dim=2)

    assert decomposition.total.shape == (2,)
    assert decomposition.aleatoric.shape == (2,)
    assert torch.allclose(decomposition.epistemic, decomposition.total - decomposition.aleatoric)


def test_decompose_registered_for_nested_embedding_samples() -> None:
    embeddings = TorchEmbedding(torch.randn(2, 3, 4, 5))
    inner = TorchEmbeddingSample(tensor=embeddings, sample_dim=2)
    outer = TorchEmbeddingSampleSample(tensor=inner, sample_dim=1)

    decomposition = decompose(outer)

    assert decomposition.total.shape == (2,)
    assert decomposition.aleatoric.shape == (2,)


def test_spectral_decomposition_rejects_weighted_samples() -> None:
    embeddings = TorchEmbedding(torch.randn(2, 3, 4))
    inner = TorchEmbeddingSample(tensor=embeddings, sample_dim=1, weights=torch.ones(3))
    outer = TorchEmbeddingSampleSample(tensor=inner, sample_dim=0)

    with pytest.raises(ValueError, match="Weighted"):
        decompose(outer)
