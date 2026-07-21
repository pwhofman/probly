"""Spectral uncertainty via Von Neumann entropy on kernel matrices.

Implements the kernel-based uncertainty computation from Walha et al.,
"Fine-Grained Uncertainty Decomposition in Large Language Models:
A Spectral Approach" (2025).

Two modes:
  - Total: Von Neumann entropy of the full response kernel matrix
  - Decomposed: aleatoric (mean per-group entropy) and epistemic
    (Holevo information = total - aleatoric)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def rbf_kernel_matrix(embeddings: NDArray[np.floating], gamma: float = 1.0) -> NDArray[np.float64]:
    """Compute the RBF (Gaussian) kernel matrix from L2-normalized embeddings.

    For unit-norm vectors: ``||e_i - e_j||^2 = 2 - 2 * (e_i . e_j)``,
    so ``K[i,j] = exp(-gamma * (2 - 2 * E @ E.T))``.

    Args:
        embeddings: Array of shape ``(n, d)`` with L2-normalized rows.
        gamma: RBF bandwidth parameter (default 1.0).

    Returns:
        Kernel matrix of shape ``(n, n)``.
    """
    similarity = embeddings @ embeddings.T
    return np.exp(-gamma * (2.0 - 2.0 * similarity))


def von_neumann_entropy(kernel: NDArray[np.floating]) -> float:
    """Compute Von Neumann entropy from a kernel matrix.

    Normalizes ``K`` to a density matrix ``rho = K / tr(K)``, then computes
    ``H_VN = -sum(lambda_i * log(lambda_i))`` from its eigenvalues.

    Args:
        kernel: Symmetric positive semi-definite kernel matrix of shape ``(n, n)``.

    Returns:
        Von Neumann entropy (non-negative).
    """
    n = kernel.shape[0]
    if n <= 1:
        return 0.0

    trace = np.trace(kernel)
    if trace < 1e-12:
        return 0.0

    rho = kernel / trace
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]

    if len(eigenvalues) == 0:
        return 0.0

    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def spectral_total_uncertainty(
    embeddings: NDArray[np.floating],
    gamma: float = 1.0,
) -> float:
    """Compute total spectral uncertainty from response embeddings.

    Builds an RBF kernel matrix and returns its Von Neumann entropy.

    Args:
        embeddings: L2-normalized embeddings of shape ``(n, d)``.
        gamma: RBF bandwidth parameter.

    Returns:
        Total uncertainty (Von Neumann entropy).
    """
    kernel = rbf_kernel_matrix(embeddings, gamma)
    return von_neumann_entropy(kernel)


@dataclass
class SpectralDecomposition:
    """Uncertainty decomposition into aleatoric and epistemic components."""

    total: float
    aleatoric: float
    epistemic: float


def spectral_decomposed_uncertainty(
    embeddings: NDArray[np.floating],
    group_sizes: list[int],
    gamma: float = 1.0,
) -> SpectralDecomposition:
    """Decompose spectral uncertainty into aleatoric and epistemic components.

    Uses the Holevo information decomposition:
      - Total = Von Neumann entropy of the full kernel
      - Aleatoric = mean Von Neumann entropy of per-group kernels
      - Epistemic = Total - Aleatoric (Holevo information)

    Args:
        embeddings: L2-normalized embeddings of shape ``(sum(group_sizes), d)``.
            Rows are ordered by group: first ``group_sizes[0]`` rows belong to
            group 0, next ``group_sizes[1]`` to group 1, etc.
        group_sizes: Number of responses per clarification group.
        gamma: RBF bandwidth parameter.

    Returns:
        Decomposition with total, aleatoric, and epistemic uncertainty.
    """
    total = spectral_total_uncertainty(embeddings, gamma)

    offset = 0
    group_entropies = []
    for size in group_sizes:
        group_emb = embeddings[offset : offset + size]
        group_entropies.append(spectral_total_uncertainty(group_emb, gamma))
        offset += size

    aleatoric = float(np.mean(group_entropies))
    epistemic = total - aleatoric

    return SpectralDecomposition(total=total, aleatoric=aleatoric, epistemic=epistemic)
