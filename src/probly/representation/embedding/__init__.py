"""Embedding representations."""

from probly.lazy_types import TORCH_VALUE

from ._common import Embedding, EmbeddingSample, EmbeddingSampleSample, create_embedding


@create_embedding.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "Embedding",
    "EmbeddingSample",
    "EmbeddingSampleSample",
    "create_embedding",
]
