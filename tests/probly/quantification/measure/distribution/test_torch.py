"""Tests for PyTorch distribution measures."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from probly.quantification.measure.distribution import (
    conditional_entropy,
    entropy,
    entropy_of_expected_value,
    mutual_information,
)
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)


def test_torch_categorical_entropy_matches_torch_distribution() -> None:
    probabilities = torch.tensor(
        [[0.25, 0.25, 0.5], [0.1, 0.2, 0.7]],
        dtype=torch.float64,
    )
    distribution = TorchCategoricalDistribution(probabilities)

    measured = entropy(distribution)
    expected = Categorical(probs=probabilities).entropy()

    assert torch.allclose(measured, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_torch_categorical_second_order_measures_match_torch(sample_axis: int) -> None:
    base_probabilities = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=torch.float64,
    )
    probabilities = torch.moveaxis(base_probabilities, 0, sample_axis)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchCategoricalDistribution(probabilities),
        sample_dim=sample_axis,
    )

    measured_entropy_of_expected = entropy_of_expected_value(sample)
    measured_conditional_entropy = conditional_entropy(sample)
    measured_mutual_information = mutual_information(sample)

    expected_mean = torch.mean(probabilities, dim=sample_axis)
    expected_entropy_of_expected = Categorical(probs=expected_mean).entropy()
    expected_conditional_entropy = torch.mean(Categorical(probs=probabilities).entropy(), dim=sample_axis)
    expected_mutual_information = torch.mean(
        kl_divergence(Categorical(probs=probabilities), Categorical(probs=expected_mean.unsqueeze(sample_axis))),
        dim=sample_axis,
    )

    assert torch.allclose(measured_entropy_of_expected, expected_entropy_of_expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(measured_conditional_entropy, expected_conditional_entropy, rtol=1e-12, atol=1e-12)
    assert torch.allclose(measured_mutual_information, expected_mutual_information, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_identity_holds_for_torch_categorical_sample(sample_axis: int) -> None:
    base_probabilities = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=torch.float64,
    )
    probabilities = torch.moveaxis(base_probabilities, 0, sample_axis)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchCategoricalDistribution(probabilities),
        sample_dim=sample_axis,
    )

    expected_entropy = entropy_of_expected_value(sample)
    decomposition_sum = conditional_entropy(sample) + mutual_information(sample)

    assert torch.allclose(expected_entropy, decomposition_sum, rtol=1e-12, atol=1e-12)
