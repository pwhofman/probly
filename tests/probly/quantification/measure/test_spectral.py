"""Tests for spectral uncertainty measures."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification.measure.spectral.torch import rbf_kernel, spectral_entropy, von_neumann_entropy
from probly.representation.embedding.torch import TorchEmbedding


def test_rbf_kernel_uses_normalized_distance_identity_case() -> None:
    embeddings = torch.eye(2)

    kernel = rbf_kernel(embeddings, gamma=0.5, sample_dim=0)

    assert torch.allclose(
        kernel, torch.tensor([[1.0, torch.exp(torch.tensor(-1.0))], [torch.exp(torch.tensor(-1.0)), 1.0]])
    )


def test_von_neumann_entropy_handles_identity_kernel() -> None:
    kernel = torch.eye(2)

    entropy = von_neumann_entropy(kernel)

    assert torch.allclose(entropy, torch.log(torch.tensor(2.0)))


def test_von_neumann_entropy_handles_singleton_kernel() -> None:
    entropy = von_neumann_entropy(torch.ones(3, 1, 1))

    assert torch.equal(entropy, torch.zeros(3))


def test_spectral_entropy_reduces_sample_axes_and_keeps_batch_shape() -> None:
    embeddings = TorchEmbedding(
        torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
            ]
        )
    )

    entropy = spectral_entropy(embeddings, sample_dim=(1, 2))

    assert entropy.shape == (2,)
    assert torch.all(entropy >= 0)
