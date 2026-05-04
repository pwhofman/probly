"""Torch-backed embedding representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, override

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.embedding._common import Embedding, EmbeddingSample, EmbeddingSampleSample
from probly.representation.sample.torch import TorchSample
from probly.representation.torch_functions import torch_average

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchEmbedding(TorchAxisProtected[torch.Tensor], Embedding[torch.Tensor]):
    """Embedding vectors with a protected trailing embedding axis.

    Shape: ``batch_shape``. The underlying ``embeddings`` tensor has shape
    ``(*batch_shape, embedding_dim)``.
    """

    embeddings: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"embeddings": 1}
    permitted_functions: ClassVar[set[Callable[..., Any]]] = {torch.mean, torch.sum, torch_average}

    def __post_init__(self) -> None:
        """Validate embedding tensor shape."""
        if not isinstance(self.embeddings, torch.Tensor):
            msg = "embeddings must be a torch tensor."
            raise TypeError(msg)
        if self.embeddings.ndim < 1:
            msg = "embeddings must have at least one dimension."
            raise ValueError(msg)
        if not torch.is_floating_point(self.embeddings):
            msg = "embeddings must be a floating point tensor."
            raise TypeError(msg)


class TorchEmbeddingSample(  # ty:ignore[conflicting-metaclass]
    EmbeddingSample[TorchEmbedding],
    TorchSample[TorchEmbedding],
):
    """A torch sample of embeddings."""

    sample_space: ClassVar[type[TorchEmbedding]] = TorchEmbedding

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


class TorchEmbeddingSampleSample(  # ty:ignore[conflicting-metaclass]
    EmbeddingSampleSample[TorchEmbedding],
    TorchSample[Any],
):
    """A torch sample of embedding samples."""

    sample_space: ClassVar[type[TorchEmbeddingSample]] = TorchEmbeddingSample

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)
