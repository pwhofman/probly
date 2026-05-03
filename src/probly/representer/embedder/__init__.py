"""Text embedding representers."""

from __future__ import annotations

from .huggingface import DEFAULT_EMBEDDING_MODEL, HFTextEmbedder

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "HFTextEmbedder",
]
