"""Tests for zero-one decomposition on PyTorch representations."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification import SecondOrderZeroOneDecomposition
from probly.quantification.measure.distribution import (
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
)
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)


def _torch_categorical_sample() -> TorchCategoricalDistributionSample:
    probabilities = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=torch.float64,
    )
    return TorchCategoricalDistributionSample(
        tensor=TorchCategoricalDistribution(probabilities),
        sample_dim=0,
    )


def test_torch_zero_one_decomposition_matches_measure_functions() -> None:
    sample = _torch_categorical_sample()

    decomposition = SecondOrderZeroOneDecomposition(sample)

    assert torch.allclose(decomposition.total, max_probability_complement_of_expected(sample), rtol=1e-12, atol=1e-12)
    assert torch.allclose(decomposition.aleatoric, expected_max_probability_complement(sample), rtol=1e-12, atol=1e-12)
    assert torch.allclose(decomposition.epistemic, max_disagreement(sample), rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        decomposition.total, decomposition.aleatoric + decomposition.epistemic, rtol=1e-12, atol=1e-12
    )


def test_torch_zero_one_decomposition_known_values() -> None:
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

    decomposition = SecondOrderZeroOneDecomposition(sample)

    assert decomposition.total.item() == pytest.approx(0.45, abs=1e-12)
    assert decomposition.aleatoric.item() == pytest.approx(0.15, abs=1e-12)
    assert decomposition.epistemic.item() == pytest.approx(0.30, abs=1e-12)


def test_torch_zero_one_decomposition_notion_access_and_types_match_backend() -> None:
    decomposition = SecondOrderZeroOneDecomposition(_torch_categorical_sample())

    total = decomposition["tu"]
    aleatoric = decomposition["au"]
    epistemic = decomposition["eu"]

    assert isinstance(total, torch.Tensor)
    assert isinstance(aleatoric, torch.Tensor)
    assert isinstance(epistemic, torch.Tensor)


def test_torch_zero_one_decomposition_caches_component_objects() -> None:
    decomposition = SecondOrderZeroOneDecomposition(_torch_categorical_sample())

    total = decomposition.total
    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic

    assert decomposition.total is total
    assert decomposition.aleatoric is aleatoric
    assert decomposition.epistemic is epistemic
    assert decomposition["tu"] is total
    assert decomposition["au"] is aleatoric
    assert decomposition["eu"] is epistemic
