"""Common code for spectral measures of embedding representations."""

from __future__ import annotations

from flextype import flexdispatch
from probly.quantification._quantification import measure
from probly.representation.embedding._common import Embedding, EmbeddingSample


@measure.register(Embedding | EmbeddingSample)
@flexdispatch
def spectral_entropy[T](
    embeddings: Embedding[T] | EmbeddingSample[Embedding[T]],
    *,
    sample_dim: int | tuple[int, ...] = -1,
    gamma: float = 1.0,
    normalized: bool = True,
    eps: float = 1e-12,
) -> T:
    """Compute spectral uncertainty as Von Neumann entropy of an RBF kernel.

    Args:
        embeddings: Embeddings with one protected trailing embedding axis.
        sample_dim: Batch axis or axes containing samples. Multiple axes are flattened.
        gamma: RBF bandwidth parameter.
        normalized: Whether embeddings are L2-normalized.
        eps: Numerical cutoff for trace and eigenvalues.

    Returns:
        Spectral entropy over the remaining batch shape.
    """
    msg = f"spectral_entropy not implemented for embeddings of type {type(embeddings)}."
    raise NotImplementedError(msg)
