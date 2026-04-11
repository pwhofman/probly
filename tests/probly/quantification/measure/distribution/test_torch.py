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

CATEGORICAL_BASES: tuple[None | float | str, ...] = (None, 2.0, "normalize")


def _base_divisor(
    base: None | float | str, num_classes: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if base is None or base == torch.e:
        return torch.tensor(1.0, dtype=dtype, device=device)
    resolved_base = float(num_classes) if base == "normalize" else float(base)
    return torch.log(torch.tensor(resolved_base, dtype=dtype, device=device))


def _tol(base: None | float | str) -> tuple[float, float]:
    if base is None:
        return 1e-12, 1e-12
    return 1e-7, 1e-7


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
def test_torch_categorical_entropy_matches_torch_distribution(base: None | float | str) -> None:
    probabilities = torch.tensor(
        [[0.25, 0.25, 0.5], [0.1, 0.2, 0.7]],
        dtype=torch.float64,
    )
    distribution = TorchCategoricalDistribution(probabilities)

    measured = entropy(distribution, base=base)
    expected_natural = Categorical(probs=probabilities).entropy()
    expected = expected_natural / _base_divisor(
        base,
        probabilities.shape[-1],
        dtype=expected_natural.dtype,
        device=expected_natural.device,
    )

    rtol, atol = _tol(base)
    assert torch.allclose(measured, expected, rtol=rtol, atol=atol)


def test_torch_categorical_entropy_normalize_maps_to_unit_interval() -> None:
    probabilities = torch.tensor(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    measured = entropy(TorchCategoricalDistribution(probabilities), base="normalize")

    assert measured[0] == pytest.approx(1.0, abs=1e-6)
    assert measured[1] == pytest.approx(0.0, abs=1e-6)
    assert torch.all(measured >= 0.0)
    assert torch.all(measured <= 1.0)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_torch_categorical_second_order_measures_match_torch(sample_axis: int, base: None | float | str) -> None:
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

    measured_entropy_of_expected = entropy_of_expected_value(sample, base=base)
    measured_conditional_entropy = conditional_entropy(sample, base=base)
    measured_mutual_information = mutual_information(sample, base=base)

    expected_mean = torch.mean(probabilities, dim=sample_axis)
    expected_entropy_of_expected_natural = Categorical(probs=expected_mean).entropy()
    expected_conditional_entropy_natural = torch.mean(Categorical(probs=probabilities).entropy(), dim=sample_axis)
    expected_mutual_information_natural = torch.mean(
        kl_divergence(Categorical(probs=probabilities), Categorical(probs=expected_mean.unsqueeze(sample_axis))),
        dim=sample_axis,
    )

    divisor = _base_divisor(
        base,
        probabilities.shape[-1],
        dtype=expected_entropy_of_expected_natural.dtype,
        device=expected_entropy_of_expected_natural.device,
    )
    expected_entropy_of_expected = expected_entropy_of_expected_natural / divisor
    expected_conditional_entropy = expected_conditional_entropy_natural / divisor
    expected_mutual_information = expected_mutual_information_natural / divisor

    rtol, atol = _tol(base)
    assert torch.allclose(measured_entropy_of_expected, expected_entropy_of_expected, rtol=rtol, atol=atol)
    assert torch.allclose(measured_conditional_entropy, expected_conditional_entropy, rtol=rtol, atol=atol)
    assert torch.allclose(measured_mutual_information, expected_mutual_information, rtol=rtol, atol=atol)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_identity_holds_for_torch_categorical_sample(sample_axis: int, base: None | float | str) -> None:
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

    expected_entropy = entropy_of_expected_value(sample, base=base)
    decomposition_sum = conditional_entropy(sample, base=base) + mutual_information(sample, base=base)

    rtol, atol = _tol(base)
    assert torch.allclose(expected_entropy, decomposition_sum, rtol=rtol, atol=atol)
