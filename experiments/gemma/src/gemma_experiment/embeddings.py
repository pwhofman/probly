"""Sentence-transformer embeddings for semantic similarity computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sentence_transformers import SentenceTransformer
import torch

from gemma_experiment.paths import CACHE_DIR

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

EmbedModel = Literal[
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]
DEFAULT_EMBED_MODEL: EmbedModel = "sentence-transformers/all-mpnet-base-v2"


class SentenceEmbedder:
    """Sentence-transformer model for dense text embeddings."""

    def __init__(self, model_name: EmbedModel = DEFAULT_EMBED_MODEL, device: str | None = None) -> None:
        """Load the sentence-transformer model."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, cache_folder=str(CACHE_DIR), device=device)

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Compute L2-normalized sentence embeddings.

        Args:
            texts: List of strings to embed.

        Returns:
            Array of shape ``(len(texts), embedding_dim)`` with unit-norm rows.
        """
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
