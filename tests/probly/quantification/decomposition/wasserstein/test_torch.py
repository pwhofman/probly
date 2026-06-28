"""Tests for the distance-based (Wasserstein) decomposition on PyTorch representations."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification import SecondOrderWassersteinDecomposition
from probly.quantification.measure.distribution import (
    expected_max_probability_complement,
    max_probability_complement_of_expected,
    min_expected_total_variation,
)
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution


def _binary_sample() -> TorchCategoricalDistributionSample:
    probabilities = torch.tensor([[0.90, 0.10], [0.50, 0.50]], dtype=torch.float64)
    return TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=0,
    )


def test_torch_wasserstein_decomposition_known_values_and_non_additive() -> None:
    decomposition = SecondOrderWassersteinDecomposition(_binary_sample())

    assert decomposition.total.item() == pytest.approx(0.3, abs=1e-9)
    assert decomposition.aleatoric.item() == pytest.approx(0.3, abs=1e-9)
    assert decomposition.epistemic.item() == pytest.approx(0.2, abs=1e-9)
    assert not torch.allclose(decomposition.total, decomposition.aleatoric + decomposition.epistemic)


def test_torch_wasserstein_decomposition_matches_measure_functions() -> None:
    sample = _binary_sample()
    decomposition = SecondOrderWassersteinDecomposition(sample)

    assert torch.allclose(decomposition.total, max_probability_complement_of_expected(sample))
    assert torch.allclose(decomposition.aleatoric, expected_max_probability_complement(sample))
    assert torch.allclose(decomposition.epistemic, min_expected_total_variation(sample))


def test_torch_wasserstein_decomposition_satisfies_axiom_a3_and_ranges() -> None:
    torch.manual_seed(0)
    logits = torch.randn(20, 8, 4, dtype=torch.float64)
    probabilities = torch.softmax(logits, dim=-1)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=1,
    )
    decomposition = SecondOrderWassersteinDecomposition(sample)
    total, aleatoric, epistemic = decomposition.total, decomposition.aleatoric, decomposition.epistemic

    assert torch.all(aleatoric <= total + 1e-9)
    assert torch.all(epistemic <= total + 1e-9)
    assert torch.all(total <= 3.0 / 4.0 + 1e-9)


def test_torch_wasserstein_decomposition_caches_components() -> None:
    decomposition = SecondOrderWassersteinDecomposition(_binary_sample())

    assert decomposition.total is decomposition.total
    assert decomposition.epistemic is decomposition.epistemic


def test_torch_wasserstein_decomposition_dirichlet_total_is_closed_form() -> None:
    distribution = TorchDirichletDistribution(torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float64))
    torch.manual_seed(0)
    decomposition = SecondOrderWassersteinDecomposition(distribution, num_samples=2000)

    assert torch.allclose(decomposition.total, torch.tensor([0.5], dtype=torch.float64))
    assert torch.all(decomposition.aleatoric >= 0.0)
    assert torch.all(decomposition.epistemic >= 0.0)
