"""Tests for entropy decomposition on PyTorch representations."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification import SecondOrderEntropyDecomposition, quantify
from probly.quantification.measure.distribution import (
    conditional_entropy,
    entropy_of_expected_value,
    mutual_information,
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


def test_quantify_dispatches_to_entropy_decomposition_for_torch_distribution_sample() -> None:
    sample = _torch_categorical_sample()

    decomposition = quantify(sample)

    assert isinstance(decomposition, SecondOrderEntropyDecomposition)


def test_torch_distribution_sample_decomposition_matches_measure_functions() -> None:
    sample = _torch_categorical_sample()

    decomposition = quantify(sample)

    assert torch.allclose(decomposition.total, entropy_of_expected_value(sample), rtol=1e-12, atol=1e-12)
    assert torch.allclose(decomposition.aleatoric, conditional_entropy(sample), rtol=1e-12, atol=1e-12)
    assert torch.allclose(decomposition.epistemic, mutual_information(sample), rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        decomposition.total, decomposition.aleatoric + decomposition.epistemic, rtol=1e-12, atol=1e-12
    )


def test_torch_decomposition_notion_access_and_types_match_backend() -> None:
    decomposition = quantify(_torch_categorical_sample())

    total = decomposition["tu"]
    aleatoric = decomposition["au"]
    epistemic = decomposition["eu"]

    assert isinstance(total, torch.Tensor)
    assert isinstance(aleatoric, torch.Tensor)
    assert isinstance(epistemic, torch.Tensor)


def test_torch_decomposition_caches_component_objects() -> None:
    decomposition = quantify(_torch_categorical_sample())

    total = decomposition.total
    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic

    assert decomposition.total is total
    assert decomposition.aleatoric is aleatoric
    assert decomposition.epistemic is epistemic
    assert decomposition["tu"] is total
    assert decomposition["au"] is aleatoric
    assert decomposition["eu"] is epistemic
