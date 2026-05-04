"""Common components for embedding representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from flextype import flexdispatch
from probly.representation.representation import Representation
from probly.representation.sample._common import RepresentationSample


class Embedding[T](Representation, ABC):
    """Abstract base class for embedding representations."""

    @property
    @abstractmethod
    def embeddings(self) -> T:
        """Get the embedding tensor."""


class EmbeddingSample[T: Embedding](RepresentationSample[T]):
    """Abstract base class for samples of embedding representations."""

    sample_space: ClassVar[type[Embedding]] = Embedding

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


class EmbeddingSampleSample[T: Embedding](RepresentationSample[EmbeddingSample[T]]):
    """Abstract base class for samples of samples of embedding representations."""

    sample_space: ClassVar[type[EmbeddingSample]] = EmbeddingSample

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


@flexdispatch
def create_embedding[T](
    embeddings: T,
) -> Embedding[T]:
    """Factory function to create an Embedding representation.

    Args:
        embeddings: The embeddings.

    Returns:
        An Embedding representation containing the given embeddings.
    """
    msg = f"create_embedding not implemented for embeddings of type {type(embeddings)}."
    raise NotImplementedError(msg)
