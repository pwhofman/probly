"""Tests for spectral uncertainty measures."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification.measure.spectral import conditional_spectral_entropy
from probly.quantification.measure.spectral.torch import rbf_kernel, spectral_entropy, von_neumann_entropy
from probly.representation.embedding.torch import TorchEmbedding, TorchEmbeddingSample, TorchEmbeddingSampleSample


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


def test_spectral_entropy_flattens_groups_of_nested_samples() -> None:
    tensor = torch.randn(2, 3, 4, 5)
    inner = TorchEmbeddingSample(tensor=TorchEmbedding(tensor), sample_dim=2)
    outer = TorchEmbeddingSampleSample(tensor=inner, sample_dim=1)

    entropy = spectral_entropy(outer)

    assert torch.allclose(entropy, spectral_entropy(TorchEmbedding(tensor), sample_dim=(1, 2)))


def test_conditional_spectral_entropy_averages_group_entropies() -> None:
    tensor = torch.randn(2, 3, 4, 5)
    inner = TorchEmbeddingSample(tensor=TorchEmbedding(tensor), sample_dim=2)
    outer = TorchEmbeddingSampleSample(tensor=inner, sample_dim=1)

    entropy = conditional_spectral_entropy(outer)

    expected = spectral_entropy(TorchEmbedding(tensor), sample_dim=2).mean(dim=1)
    assert entropy.shape == (2,)
    assert torch.allclose(entropy, expected)


def test_rbf_kernel_unnormalized_matches_pairwise_distances() -> None:
    embeddings = torch.randn(5, 3)

    kernel = rbf_kernel(embeddings, gamma=0.7, sample_dim=0, normalized=False)

    expected = torch.exp(-0.7 * torch.cdist(embeddings, embeddings) ** 2)
    assert torch.allclose(kernel, expected, atol=1e-6)


def test_rbf_kernel_rejects_non_positive_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        rbf_kernel(torch.eye(2), gamma=0.0, sample_dim=0)


def test_rbf_kernel_rejects_non_tensor_embeddings() -> None:
    with pytest.raises(TypeError, match="TorchEmbedding"):
        rbf_kernel([[1.0, 0.0], [0.0, 1.0]], sample_dim=0)


def test_von_neumann_entropy_is_zero_for_identical_samples() -> None:
    # Identical unit vectors give an all-ones kernel, a rank-one density matrix and hence a pure state.
    entropy = von_neumann_entropy(torch.ones(3, 3))

    assert torch.allclose(entropy, torch.zeros(()), atol=1e-5)


def test_von_neumann_entropy_is_zero_for_vanishing_trace() -> None:
    entropy = von_neumann_entropy(torch.zeros(2, 2))

    assert torch.equal(entropy, torch.zeros(()))


@pytest.mark.parametrize(
    ("kernel", "eps", "error"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], 1e-12, TypeError),
        (torch.ones(2, 3), 1e-12, ValueError),
        (torch.ones(3), 1e-12, ValueError),
        (torch.eye(2), -1.0, ValueError),
    ],
)
def test_von_neumann_entropy_validates_inputs(kernel: object, eps: float, error: type[Exception]) -> None:
    with pytest.raises(error):
        von_neumann_entropy(kernel, eps=eps)


def test_spectral_entropy_uses_sample_axis_of_embedding_samples() -> None:
    tensor = torch.randn(2, 3, 4)
    sample = TorchEmbeddingSample(tensor=TorchEmbedding(tensor), sample_dim=1)

    entropy = spectral_entropy(sample)

    assert entropy.shape == (2,)
    assert torch.allclose(entropy, spectral_entropy(TorchEmbedding(tensor), sample_dim=1))


def test_conditional_spectral_entropy_rejects_group_axis_equal_to_sample_axis() -> None:
    inner = TorchEmbeddingSample(tensor=TorchEmbedding(torch.randn(2, 3, 4)), sample_dim=1)
    outer = TorchEmbeddingSampleSample(tensor=inner, sample_dim=1)

    with pytest.raises(ValueError, match="group axis"):
        conditional_spectral_entropy(outer)


def test_spectral_measures_reject_unsupported_types() -> None:
    with pytest.raises(NotImplementedError, match="spectral_entropy"):
        spectral_entropy(object())
    with pytest.raises(NotImplementedError, match="conditional_spectral_entropy"):
        conditional_spectral_entropy(object())
