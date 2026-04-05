"""Tests for torch categorical distribution representation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.distribution import create_categorical_distribution
from probly.representation.distribution.torch_categorical import TorchTensorCategoricalDistribution
from probly.representation.sample.torch import TorchTensorSample


def test_create_categorical_distribution_from_torch_tensor() -> None:
    probabilities = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)

    dist = create_categorical_distribution(probabilities)

    assert isinstance(dist, TorchTensorCategoricalDistribution)
    assert torch.equal(dist.probabilities, probabilities)


def test_accepts_relative_non_negative_probabilities() -> None:
    probabilities = torch.tensor([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]], dtype=torch.float64)

    dist = TorchTensorCategoricalDistribution(probabilities=probabilities)

    assert dist.shape == (2,)
    assert dist.num_classes == 3


def test_rejects_negative_relative_probabilities() -> None:
    probabilities = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="non-negative"):
        TorchTensorCategoricalDistribution(probabilities=probabilities)


def test_entropy_normalizes_relative_probabilities() -> None:
    probabilities = torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float64)
    dist = TorchTensorCategoricalDistribution(probabilities=probabilities)

    normalized = probabilities / torch.sum(probabilities, dim=-1, keepdim=True)
    expected = -torch.sum(normalized * torch.log(normalized), dim=-1)

    assert torch.allclose(dist.entropy, expected)


def test_entropy_bernoulli_formula() -> None:
    probabilities = torch.tensor([[0.25], [0.5], [0.75]], dtype=torch.float64)
    dist = TorchTensorCategoricalDistribution(probabilities=probabilities)

    p = probabilities[:, 0]
    expected = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    assert torch.allclose(dist.entropy, expected)


def test_sampling_relative_probabilities_matches_normalized_distribution() -> None:
    probabilities = torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float64)
    dist = TorchTensorCategoricalDistribution(probabilities=probabilities)
    rng = torch.Generator().manual_seed(0)

    sample = dist.sample(num_samples=30_000, rng=rng)

    assert isinstance(sample, TorchTensorSample)
    assert sample.sample_dim == 0
    assert sample.tensor.shape == (30_000, 1)
    assert sample.tensor.dtype == torch.int64

    counts = torch.bincount(sample.tensor[:, 0], minlength=dist.num_classes).to(dtype=torch.float64)
    frequencies = counts / torch.sum(counts)
    expected = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)

    assert torch.allclose(frequencies, expected, atol=0.02)


def test_sampling_bernoulli_produces_binary_samples_with_correct_mean() -> None:
    probabilities = torch.tensor([[0.3]], dtype=torch.float64)
    dist = TorchTensorCategoricalDistribution(probabilities=probabilities)
    rng = torch.Generator().manual_seed(1)

    sample = dist.sample(num_samples=40_000, rng=rng)

    assert isinstance(sample, TorchTensorSample)
    assert sample.tensor.shape == (40_000, 1)
    assert torch.all((sample.tensor == 0) | (sample.tensor == 1))
    assert float(torch.mean(sample.tensor.to(dtype=torch.float64))) == pytest.approx(0.3, abs=0.02)


def test_numpy_array_interop() -> None:
    probabilities = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
    dist = TorchTensorCategoricalDistribution(probabilities=probabilities)

    array = np.asarray(dist, dtype=np.float32)

    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    np.testing.assert_allclose(array, np.array([[0.2, 0.8]], dtype=np.float32))


def test_getitem_cannot_index_class_axis_directly() -> None:
    probabilities = torch.arange(24, dtype=torch.float64).reshape((2, 3, 4)) + 1.0
    dist = TorchTensorCategoricalDistribution(probabilities=probabilities)

    with pytest.raises(IndexError):
        _ = dist[:, :, 0]


def test_expand_dims_last_inserts_before_class_axis() -> None:
    probabilities = torch.arange(24, dtype=torch.float64).reshape((2, 3, 4)) + 1.0
    dist = TorchTensorCategoricalDistribution(probabilities=probabilities)

    expanded = torch.unsqueeze(dist, dim=-1)

    assert isinstance(expanded, TorchTensorCategoricalDistribution)
    assert expanded.shape == (2, 3, 1)
    assert expanded.probabilities.shape == (2, 3, 1, 4)


def test_reshape_inserts_before_class_axis() -> None:
    probabilities = torch.arange(24, dtype=torch.float64).reshape((2, 3, 4)) + 1.0
    dist = TorchTensorCategoricalDistribution(probabilities=probabilities)

    reshaped = torch.reshape(dist, (6, 1))

    assert isinstance(reshaped, TorchTensorCategoricalDistribution)
    assert reshaped.shape == (6, 1)
    assert reshaped.probabilities.shape == (6, 1, 4)
