"""Embedding representations."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import Embedding, EmbeddingSample, EmbeddingSampleSample, create_embedding


@create_embedding.register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "Embedding",
    "EmbeddingSample",
    "EmbeddingSampleSample",
    "create_embedding",
]
