"""Tests for torch embedding representations."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.representation.embedding import TorchEmbedding, TorchEmbeddingSample, TorchEmbeddingSampleSample


def test_torch_embedding_protects_trailing_embedding_axis() -> None:
    embeddings = TorchEmbedding(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))

    assert embeddings.shape == (2, 3)
    assert embeddings.protected_shape == (4,)
    assert embeddings[0].shape == (3,)
    assert embeddings.reshape(6).embeddings.shape == (6, 4)


def test_torch_embedding_sample_wrappers_preserve_axes() -> None:
    embeddings = TorchEmbedding(torch.randn(2, 3, 4))
    inner = TorchEmbeddingSample(tensor=embeddings, sample_dim=1)
    outer = TorchEmbeddingSampleSample(tensor=inner, sample_dim=0)

    assert inner.sample_size == 3
    assert outer.sample_size == 2
    assert outer.tensor.tensor is embeddings


def test_torch_embedding_rejects_non_floating_tensor() -> None:
    with pytest.raises(TypeError, match="floating"):
        TorchEmbedding(torch.ones(2, 3, dtype=torch.long))
