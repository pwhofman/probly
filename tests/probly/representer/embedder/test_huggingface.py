"""Tests for Hugging Face text embedding representers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.embedding.torch import TorchEmbedding, TorchEmbeddingSample, TorchEmbeddingSampleSample
from probly.representation.text_generation.torch import (
    TorchTextGeneration,
    TorchTextGenerationSample,
    TorchTextGenerationSampleSample,
)

pytest.importorskip("sentence_transformers")
from probly.representer.embedder.huggingface import DEFAULT_EMBEDDING_MODEL, HFTextEmbedder


class FakeEmbeddingModel:
    """Fake sentence embedding model with deterministic outputs."""

    def __init__(self) -> None:
        """Initialize call recording."""
        self.calls: list[dict[str, object]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> torch.Tensor:
        """Encode each string as simple length and index features."""
        self.calls.append({"sentences": sentences, **kwargs})
        return torch.tensor(
            [[float(len(sentence)), float(index)] for index, sentence in enumerate(sentences)],
            dtype=torch.float32,
        )


class FakeNumpyEmbeddingModel:
    """Fake model that returns NumPy arrays despite convert_to_tensor=True."""

    def __init__(self) -> None:
        """Initialize call recording."""
        self.calls: list[dict[str, object]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append({"sentences": sentences, **kwargs})
        return np.asarray([[len(sentence), index] for index, sentence in enumerate(sentences)], dtype=np.float32)


def test_embedder_embeds_raw_text_generation_as_tensor() -> None:
    model = FakeEmbeddingModel()
    generation = TorchTextGeneration(
        text=np.asarray([["alpha", "b"], ["gamma", "delta"]], dtype=object),
        log_likelihood=torch.zeros((2, 2)),
    )

    embeddings = HFTextEmbedder(model, batch_size=7, normalize_embeddings=False)(generation)

    assert isinstance(embeddings, TorchEmbedding)
    assert embeddings.shape == (2, 2)
    assert torch.equal(
        embeddings.embeddings,
        torch.tensor(
            [
                [[5.0, 0.0], [1.0, 1.0]],
                [[5.0, 2.0], [5.0, 3.0]],
            ]
        ),
    )
    assert model.calls == [
        {
            "sentences": ["alpha", "b", "gamma", "delta"],
            "batch_size": 7,
            "normalize_embeddings": False,
            "convert_to_tensor": True,
        }
    ]


def test_embedder_embeds_text_generation_sample_without_wrapping_output() -> None:
    model = FakeEmbeddingModel()
    generation = TorchTextGeneration(
        text=np.asarray([["first", "second", "third"]], dtype=object),
        log_likelihood=torch.zeros((1, 3)),
    )
    sample = TorchTextGenerationSample(tensor=generation, sample_dim=1, weights=torch.tensor([0.2, 0.3, 0.5]))

    embeddings = HFTextEmbedder(model)(sample)

    assert isinstance(embeddings, TorchEmbeddingSample)
    assert embeddings.shape == (1, 3)
    assert embeddings.sample_dim == 1
    assert torch.equal(embeddings.weights, torch.tensor([0.2, 0.3, 0.5]))
    assert model.calls[0]["sentences"] == ["first", "second", "third"]


def test_embedder_preserves_nested_text_generation_sample_wrapping() -> None:
    model = FakeEmbeddingModel()
    generation = TorchTextGeneration(
        text=np.asarray([["a", "bb"], ["ccc", "dddd"]], dtype=object),
        log_likelihood=torch.zeros((2, 2)),
    )
    inner_sample = TorchTextGenerationSample(tensor=generation, sample_dim=1)
    outer_sample = TorchTextGenerationSampleSample(
        tensor=inner_sample,
        sample_dim=0,
        weights=torch.tensor([0.25, 0.75]),
    )

    embeddings = HFTextEmbedder(model)(outer_sample)

    assert isinstance(embeddings, TorchEmbeddingSampleSample)
    assert embeddings.sample_dim == 0
    assert torch.equal(embeddings.weights, torch.tensor([0.25, 0.75]))
    assert isinstance(embeddings.tensor, TorchEmbeddingSample)
    assert embeddings.tensor.sample_dim == 1
    assert embeddings.tensor.tensor.shape == (2, 2)
    assert embeddings.tensor.tensor.protected_shape == (2,)


def test_embedder_coerces_numpy_embeddings_to_torch_tensor() -> None:
    generation = TorchTextGeneration(
        text=np.asarray(["only"], dtype=object),
        log_likelihood=torch.zeros(1),
    )

    embeddings = HFTextEmbedder(FakeNumpyEmbeddingModel())(generation)

    assert isinstance(embeddings, TorchEmbedding)
    assert embeddings.dtype == torch.float32
    assert torch.equal(embeddings.embeddings, torch.tensor([[4.0, 0.0]]))


def test_embedder_rejects_empty_generation() -> None:
    generation = TorchTextGeneration(
        text=np.asarray([], dtype=object),
        log_likelihood=torch.empty(0),
    )

    with pytest.raises(ValueError, match="empty"):
        HFTextEmbedder(FakeEmbeddingModel())(generation)


def test_embedder_rejects_embedding_shape_mismatch() -> None:
    class BadEmbeddingModel:
        def encode(self, sentences: list[str], **kwargs: object) -> torch.Tensor:
            del sentences, kwargs
            return torch.zeros((2, 3))

    generation = TorchTextGeneration(
        text=np.asarray(["one"], dtype=object),
        log_likelihood=torch.zeros(1),
    )

    with pytest.raises(ValueError, match="returned 2 embeddings"):
        HFTextEmbedder(BadEmbeddingModel())(generation)


def test_from_model_name_loads_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    sentence_transformers = pytest.importorskip("sentence_transformers")
    created: list[dict[str, object]] = []
    model = FakeEmbeddingModel()

    class FakeSentenceTransformer:
        def __new__(cls, model_name: str, **kwargs: object) -> FakeEmbeddingModel:
            del cls
            created.append({"model_name": model_name, **kwargs})
            return model

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)

    embedder = HFTextEmbedder.from_model_name(
        cache_dir="probly-cache",
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        batch_size=5,
        normalize_embeddings=False,
        encode_kwargs={"show_progress_bar": False},
    )

    assert embedder.model is model
    assert embedder.batch_size == 5
    assert embedder.normalize_embeddings is False
    assert embedder.encode_kwargs == {"show_progress_bar": False}
    assert created == [
        {
            "model_name": DEFAULT_EMBEDDING_MODEL,
            "device": "cpu",
            "trust_remote_code": True,
            "cache_folder": "probly-cache",
        }
    ]


def test_embedder_rejects_invalid_input_type() -> None:
    with pytest.raises(TypeError, match="TorchTextGeneration"):
        HFTextEmbedder(FakeEmbeddingModel())(SimpleNamespace(text=np.asarray(["nope"], dtype=object)))
