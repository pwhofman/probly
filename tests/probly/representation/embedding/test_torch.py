"""Tests for torch embedding representations."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.representation.embedding._common import EmbeddingSample
from probly.representation.embedding.torch import TorchEmbedding, TorchEmbeddingSample, TorchEmbeddingSampleSample
from probly.representation.sample._common import ListSample


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


def test_nested_sample_accepts_generic_inner_samples_and_preserves_outer_weights() -> None:
    class ListEmbeddingSample(ListSample[TorchEmbedding], EmbeddingSample[TorchEmbedding]):
        pass

    values = [TorchEmbedding(torch.full((2, 4), float(i))) for i in range(3)]
    inner = ListEmbeddingSample(values)
    outer = TorchEmbeddingSampleSample.from_iterable([inner, inner], weights=[0.25, 0.75])

    assert isinstance(outer.tensor, TorchEmbeddingSample)
    assert outer.sample_size == 2
    assert outer.tensor.sample_size == 3
    torch.testing.assert_close(outer.weights, torch.tensor([0.25, 0.75]))
    assert outer.tensor.weights is None
    weighted_inner = ListEmbeddingSample(values, weights=[0.2, 0.3, 0.5])
    with pytest.raises(ValueError, match="Weighted samples do not support stack"):
        TorchEmbeddingSampleSample.from_iterable([weighted_inner, weighted_inner])


def test_embedding_sample_normalizes_negative_axis() -> None:
    sample = TorchEmbeddingSample(TorchEmbedding(torch.ones(2, 3, 4)), sample_dim=-1)
    assert sample.sample_dim == 1
    assert sample.sample_size == 3


def test_embedding_sample_to_device_preserves_weights_and_rejects_streams() -> None:
    sample = TorchEmbeddingSample(
        TorchEmbedding(torch.ones(2, 3, 4)), sample_dim=1, weights=torch.tensor([0.2, 0.3, 0.5])
    )
    moved = sample.to_device("cpu")
    assert moved.device == torch.device("cpu")
    torch.testing.assert_close(moved.weights, sample.weights)
    with pytest.raises(NotImplementedError, match="stream"):
        sample.to_device("cpu", stream=1)
