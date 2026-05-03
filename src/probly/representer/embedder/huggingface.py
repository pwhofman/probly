"""Text embedding representers backed by Hugging Face sentence-transformers models."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Protocol, Self

import torch

from probly.representation.text_generation import TorchTextGeneration, TorchTextGenerationSample
from probly.representer._representer import Representer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

type TextEmbedInput = TorchTextGeneration | TorchTextGenerationSample


class SentenceTransformerLike(Protocol):
    """Protocol for sentence-transformer compatible embedding models."""

    def encode(self, sentences: list[str], **kwargs: object) -> object:
        """Embed a list of sentences."""


class HFTextEmbedder(Representer[Any, Any, torch.Tensor, Any]):
    """Embed text generations using a Hugging Face sentence-transformers model."""

    model: SentenceTransformerLike
    batch_size: int
    normalize_embeddings: bool
    encode_kwargs: dict[str, object]

    def __init__(
        self,
        model: SentenceTransformerLike,
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        encode_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the text embedder.

        Args:
            model: Sentence-transformer compatible embedding model.
            batch_size: Number of texts to embed in one model call.
            normalize_embeddings: Whether sentence-transformers should L2-normalize embeddings.
            encode_kwargs: Additional keyword arguments forwarded to ``model.encode``.
        """
        if batch_size <= 0:
            msg = "batch_size must be positive."
            raise ValueError(msg)

        self.model = model
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.encode_kwargs = dict(encode_kwargs or {})

    @classmethod
    def from_model_name(
        cls,
        model_name: str | None = None,
        *,
        cache_dir: str | PathLike[str] | None = None,
        model_kwargs: Mapping[str, object] | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        encode_kwargs: Mapping[str, object] | None = None,
    ) -> Self:
        """Load a sentence-transformers model by name and initialize the embedder.

        Args:
            model_name: Hugging Face model name or local path. Defaults to ``DEFAULT_EMBEDDING_MODEL``.
            cache_dir: Optional Hugging Face cache directory.
            model_kwargs: Additional keyword arguments forwarded to ``SentenceTransformer``.
            batch_size: Number of texts to embed in one model call.
            normalize_embeddings: Whether sentence-transformers should L2-normalize embeddings.
            encode_kwargs: Additional keyword arguments forwarded to ``model.encode``.

        Returns:
            A text embedder backed by the loaded sentence-transformers model.
        """
        resolved_model_name = DEFAULT_EMBEDDING_MODEL if model_name is None else model_name
        kwargs = dict(model_kwargs or {})
        if cache_dir is not None:
            kwargs["cache_folder"] = str(cache_dir)

        sentence_transformers = importlib.import_module("sentence_transformers")
        sentence_transformer = sentence_transformers.SentenceTransformer
        model = sentence_transformer(resolved_model_name, **kwargs)
        return cls(
            model=model,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            encode_kwargs=encode_kwargs,
        )

    @property
    def predictor(self) -> SentenceTransformerLike:
        """The underlying embedding model."""
        return self.model

    def _embed_text(self, generation: TorchTextGeneration) -> torch.Tensor:
        text = generation.text
        if text.size == 0:
            msg = "Cannot embed an empty text generation."
            raise ValueError(msg)

        flat_text = text.reshape(-1).astype(str, copy=False).tolist()
        embeddings = self.model.encode(
            flat_text,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=True,
            **self.encode_kwargs,
        )
        tensor = torch.as_tensor(embeddings)
        if not torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim != 2:
            msg = f"Embedding model must return a 2D tensor, got shape {tuple(tensor.shape)}."
            raise ValueError(msg)
        if tensor.shape[0] != len(flat_text):
            msg = f"Embedding model returned {tensor.shape[0]} embeddings for {len(flat_text)} texts."
            raise ValueError(msg)

        return tensor.reshape((*generation.shape, tensor.shape[-1]))

    def represent(self, generation: TextEmbedInput) -> torch.Tensor:
        """Embed text generations.

        Args:
            generation: Text generation representation or sample.

        Returns:
            Embeddings with shape ``(*generation.shape, embedding_dim)``.
        """
        if isinstance(generation, TorchTextGenerationSample):
            return self._embed_text(generation.tensor)

        if not isinstance(generation, TorchTextGeneration):
            msg = "generation must be a TorchTextGeneration or TorchTextGenerationSample."
            raise TypeError(msg)
        return self._embed_text(generation)
