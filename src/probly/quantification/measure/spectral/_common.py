"""Common code for spectral measures of embedding representations."""

from __future__ import annotations

from flextype import flexdispatch

from probly.quantification._quantification import measure
from probly.representation.embedding._common import Embedding, EmbeddingSample, EmbeddingSampleSample


@measure.register(Embedding | EmbeddingSample)
@flexdispatch
def spectral_entropy[T](
    embeddings: Embedding[T] | EmbeddingSample[Embedding[T]] | EmbeddingSampleSample[Embedding[T]],
    *,
    sample_dim: int | tuple[int, ...] = -1,
    gamma: float = 1.0,
    normalized: bool = True,
    eps: float = 1e-12,
) -> T:
    """Compute spectral uncertainty as Von Neumann entropy of an RBF kernel :cite:`walhaFineGrainedUncertainty2026`.

    Args:
        embeddings: Embeddings with one protected trailing embedding axis. Samples carry their own sample axes; for
            a sample of embedding samples, the groups and the samples within them are flattened into one set.
        sample_dim: Batch axis or axes containing samples. Multiple axes are flattened.
        gamma: RBF bandwidth parameter.
        normalized: Whether embeddings are L2-normalized.
        eps: Numerical cutoff for trace and eigenvalues.

    Returns:
        Spectral entropy over the remaining batch shape.
    """
    msg = f"spectral_entropy not implemented for embeddings of type {type(embeddings)}."
    raise NotImplementedError(msg)


@flexdispatch
def conditional_spectral_entropy[T](
    embeddings: EmbeddingSampleSample[Embedding[T]],
    *,
    gamma: float = 1.0,
    normalized: bool = True,
    eps: float = 1e-12,
) -> T:
    """Compute the mean spectral entropy within groups of embedding samples :cite:`walhaFineGrainedUncertainty2026`.

    The outer sample holds the groups, such as clarifications of a prompt, and each inner sample the responses within
    a group. The Von Neumann entropy of each group's kernel is averaged over the groups.

    Args:
        embeddings: Sample of embedding samples.
        gamma: RBF bandwidth parameter.
        normalized: Whether embeddings are L2-normalized.
        eps: Numerical cutoff for trace and eigenvalues.

    Returns:
        Mean spectral entropy over the remaining batch shape.
    """
    msg = f"conditional_spectral_entropy not implemented for embeddings of type {type(embeddings)}."
    raise NotImplementedError(msg)
