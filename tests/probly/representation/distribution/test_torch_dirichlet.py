"""Tests for torch-based Dirichlet distribution representation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.predictor import to_single_prediction  # noqa: E402
from probly.representation.distribution import create_dirichlet_distribution_from_alphas  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: E402
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: E402
from probly.representation.sample.torch import TorchSample  # noqa: E402


def test_torch_dirichlet_initialization_valid() -> None:
    alphas = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0]])

    distribution = TorchDirichletDistribution(alphas)

    assert distribution.alphas is alphas
    assert distribution.shape == (2,)
    assert distribution.ndim == 1
    assert distribution.size() == torch.Size([2])


def test_torch_dirichlet_factory_from_alphas() -> None:
    alphas = torch.tensor([[1.0, 2.0, 3.0]])

    distribution = create_dirichlet_distribution_from_alphas(alphas)

    assert isinstance(distribution, TorchDirichletDistribution)
    assert distribution.alphas is alphas


@pytest.mark.parametrize("invalid_value", [0.0, -0.1, -5.0])
def test_torch_dirichlet_raises_on_non_positive_alphas(invalid_value: float) -> None:
    alphas = torch.tensor([1.0, invalid_value, 2.0])

    with pytest.raises(ValueError, match="alphas must be strictly positive"):
        TorchDirichletDistribution(alphas)


def test_torch_dirichlet_canonical_element_is_expected_categorical_distribution() -> None:
    distribution = TorchDirichletDistribution(torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0]]))

    single = to_single_prediction(distribution)

    assert isinstance(single, TorchCategoricalDistribution)
    assert torch.allclose(single.probabilities, torch.tensor([[1 / 6, 2 / 6, 3 / 6], [2 / 8, 2 / 8, 4 / 8]]))


def test_torch_dirichlet_sample_returns_categorical_distribution_sample() -> None:
    distribution = TorchDirichletDistribution(torch.tensor([1.0, 2.0, 3.0]))

    sample = distribution.sample(num_samples=4)

    assert isinstance(sample, TorchSample)
    assert isinstance(sample.tensor, TorchCategoricalDistribution)
    assert sample.tensor.unnormalized_probabilities.shape == (4, 3)
    assert sample.sample_axis == 0
    assert torch.allclose(sample.tensor.probabilities.sum(dim=-1), torch.ones(4))


def test_torch_dirichlet_torch_operations_preserve_protected_class_axis() -> None:
    alphas = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1.0
    distribution = TorchDirichletDistribution(alphas)

    meaned = torch.mean(distribution, dim=0)

    assert isinstance(meaned, TorchDirichletDistribution)
    assert meaned.shape == (3,)
    assert meaned.alphas.shape == (3, 4)
    assert torch.allclose(meaned.alphas, torch.mean(alphas, dim=0))
