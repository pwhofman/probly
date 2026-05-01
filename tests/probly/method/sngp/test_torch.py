"""Tests for torch SNGP quantification dispatch."""

from __future__ import annotations

import pytest

from probly.method.sngp import sngp
from probly.quantification import decompose, measure, quantify
from probly.quantification.decomposition.entropy import SecondOrderEntropyDecomposition
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representer import representer

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


def _sngp_sample() -> TorchCategoricalDistributionSample:
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 3),
    )
    predictor = sngp(model, num_inducing=128)
    return representer(predictor, num_samples=3).represent(torch.ones(2, 2))


def test_sngp_representer_returns_categorical_sample() -> None:
    sample = _sngp_sample()

    assert isinstance(sample, TorchCategoricalDistributionSample)


def test_decompose_dispatches_sngp_sample_to_entropy_decomposition() -> None:
    sample = _sngp_sample()

    decomposition = decompose(sample)

    assert isinstance(decomposition, SecondOrderEntropyDecomposition)


def test_quantify_dispatches_sngp_sample_to_entropy_decomposition() -> None:
    sample = _sngp_sample()

    quantification = quantify(sample)

    assert isinstance(quantification, SecondOrderEntropyDecomposition)


def test_sngp_decomposition_contains_all_uncertainties() -> None:
    sample = _sngp_sample()

    decomposition = decompose(sample)

    assert hasattr(decomposition, "aleatoric")
    assert hasattr(decomposition, "epistemic")
    assert hasattr(decomposition, "total")

    assert torch.allclose(decomposition.total, decomposition.aleatoric + decomposition.epistemic)


def test_measure_sngp_sample_returns_total_uncertainty() -> None:
    sample = _sngp_sample()

    uncertainty = measure(sample)
    decomposition = decompose(sample)

    assert torch.allclose(uncertainty, decomposition.total)
    assert uncertainty.shape == (2,)
