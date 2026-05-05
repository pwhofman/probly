"""Tests for torch-based mixture distribution representation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.representation.distribution import MixtureDistribution  # noqa: E402
from probly.representation.distribution.torch_categorical import (  # noqa: E402
    TorchCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: E402
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution  # noqa: E402
from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: E402
from probly.representation.sample.torch import TorchSample  # noqa: E402


def test_torch_mixture_initialization_valid() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(2, 3), var=torch.ones(2, 3))
    weights = torch.tensor([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]])

    distribution = TorchMixtureDistribution(components=components, mixture_weights=weights)

    assert isinstance(distribution, MixtureDistribution)
    assert distribution.type == "mixture"
    assert distribution.shape == (2,)
    assert distribution.ndim == 1
    torch.testing.assert_close(distribution.normalized_mixture_weights.sum(dim=-1), torch.ones(2))


def test_torch_mixture_rejects_shape_mismatch() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(2, 3), var=torch.ones(2, 3))
    weights = torch.ones(2, 4)

    with pytest.raises(ValueError, match="components shape must match mixture_weights shape"):
        TorchMixtureDistribution(components=components, mixture_weights=weights)


def test_torch_mixture_rejects_invalid_weights() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(3), var=torch.ones(3))

    with pytest.raises(ValueError, match="non-negative"):
        TorchMixtureDistribution(components=components, mixture_weights=torch.tensor([1.0, -1.0, 1.0]))

    with pytest.raises(ValueError, match="positive sums"):
        TorchMixtureDistribution(components=components, mixture_weights=torch.zeros(3))


def test_torch_mixture_indexing_preserves_component_axis() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(2, 3), var=torch.ones(2, 3))
    distribution = TorchMixtureDistribution(components=components, mixture_weights=torch.ones(2, 3))

    sliced = distribution[0]

    assert isinstance(sliced, TorchMixtureDistribution)
    assert sliced.shape == ()
    assert sliced.components.shape == (3,)
    assert sliced.mixture_weights.shape == (3,)

    with pytest.raises(IndexError):
        _ = distribution[:, 0]


def test_torch_mixture_reshape_inserts_before_component_axis() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(2, 3), var=torch.ones(2, 3))
    distribution = TorchMixtureDistribution(components=components, mixture_weights=torch.ones(2, 3))

    reshaped = torch.reshape(distribution, (1, 2))

    assert isinstance(reshaped, TorchMixtureDistribution)
    assert reshaped.shape == (1, 2)
    assert reshaped.components.shape == (1, 2, 3)
    assert reshaped.mixture_weights.shape == (1, 2, 3)


def test_torch_mixture_sampling_categorical_components_matches_mixture_weights() -> None:
    components = TorchProbabilityCategoricalDistribution(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    distribution = TorchMixtureDistribution(components=components, mixture_weights=torch.tensor([1.0, 3.0]))
    rng = torch.Generator().manual_seed(0)

    sample = distribution.sample(num_samples=30_000, rng=rng)

    assert isinstance(sample, TorchSample)
    assert sample.sample_axis == 0
    assert sample.tensor.shape == (30_000,)
    assert sample.tensor.dtype == torch.int64

    counts = torch.bincount(sample.tensor, minlength=2).to(dtype=torch.float64)
    frequencies = counts / torch.sum(counts)
    torch.testing.assert_close(frequencies, torch.tensor([0.25, 0.75], dtype=torch.float64), atol=0.02, rtol=0.0)


def test_torch_mixture_sampling_dirichlet_components_preserves_class_axis() -> None:
    components = TorchDirichletDistribution(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    distribution = TorchMixtureDistribution(components=components, mixture_weights=torch.tensor([1.0, 1.0]))

    sample = distribution.sample(num_samples=4)

    assert isinstance(sample, TorchSample)
    assert isinstance(sample.tensor, TorchCategoricalDistribution)
    assert sample.tensor.shape == (4,)
    assert sample.tensor.probabilities.shape == (4, 3)
    assert sample.sample_axis == 0
    torch.testing.assert_close(sample.tensor.probabilities.sum(dim=-1), torch.ones(4))
