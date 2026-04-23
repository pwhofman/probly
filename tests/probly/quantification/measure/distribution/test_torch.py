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
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
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


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_torch_sample_zero_one_measures_match_manual(sample_axis: int) -> None:
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

    measured_total = max_probability_complement_of_expected(sample)
    measured_aleatoric = expected_max_probability_complement(sample)
    measured_epistemic = max_disagreement(sample)

    expected_mean = torch.mean(probabilities, dim=sample_axis)
    expected_total = 1.0 - torch.max(expected_mean, dim=-1).values
    expected_aleatoric = torch.mean(1.0 - torch.max(probabilities, dim=-1).values, dim=sample_axis)
    bma_argmax = torch.argmax(expected_mean, dim=-1).unsqueeze(sample_axis).unsqueeze(-1)
    per_sample_bma_prob = torch.take_along_dim(probabilities, bma_argmax, dim=-1).squeeze(-1)
    expected_epistemic = torch.mean(torch.max(probabilities, dim=-1).values - per_sample_bma_prob, dim=sample_axis)

    assert torch.allclose(measured_total, expected_total, rtol=1e-12, atol=1e-12)
    assert torch.allclose(measured_aleatoric, expected_aleatoric, rtol=1e-12, atol=1e-12)
    assert torch.allclose(measured_epistemic, expected_epistemic, rtol=1e-12, atol=1e-12)


def test_torch_sample_zero_one_known_values() -> None:
    probabilities = torch.tensor(
        [
            [0.90, 0.10],
            [0.20, 0.80],
        ],
        dtype=torch.float64,
    )
    sample = TorchCategoricalDistributionSample(
        tensor=TorchCategoricalDistribution(probabilities),
        sample_dim=0,
    )

    assert max_probability_complement_of_expected(sample).item() == pytest.approx(0.45, abs=1e-12)
    assert expected_max_probability_complement(sample).item() == pytest.approx(0.15, abs=1e-12)
    assert max_disagreement(sample).item() == pytest.approx(0.30, abs=1e-12)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_zero_one_identity_holds_for_torch_categorical_sample(sample_axis: int) -> None:
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

    total = max_probability_complement_of_expected(sample)
    aleatoric = expected_max_probability_complement(sample)
    epistemic = max_disagreement(sample)

    assert torch.allclose(total, aleatoric + epistemic, rtol=1e-12, atol=1e-12)
