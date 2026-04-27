"""Tests for torch HetNets quantification dispatch."""

from __future__ import annotations

import pytest

from probly.method.het_nets import het_nets
from probly.quantification import decompose, measure, quantify
from probly.quantification.decomposition.entropy import LabelNoiseEntropyDecomposition
from probly.representation.het_nets import HetNetsRepresentation
from probly.representer import representer

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


def _hetnets_sample() -> HetNetsRepresentation:
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 3),
    )
    predictor = het_nets(model, num_factors=2, predictor_type="logit_classifier")
    return representer(predictor, num_samples=3).represent(torch.ones(2, 2))


def test_hetnets_representer_marks_sample_as_hetnets_representation() -> None:
    sample = _hetnets_sample()

    assert isinstance(sample, HetNetsRepresentation)


def test_decompose_dispatches_hetnets_sample_to_label_noise_decomposition() -> None:
    sample = _hetnets_sample()

    decomposition = decompose(sample)

    assert isinstance(decomposition, LabelNoiseEntropyDecomposition)


def test_quantify_dispatches_hetnets_sample_to_label_noise_decomposition() -> None:
    sample = _hetnets_sample()

    quantification = quantify(sample)

    assert isinstance(quantification, LabelNoiseEntropyDecomposition)


def test_hetnets_decomposition_is_aleatoric_only() -> None:
    sample = _hetnets_sample()

    decomposition = decompose(sample)

    aleatoric = decomposition.aleatoric

    assert torch.allclose(decomposition["au"], aleatoric)
    assert aleatoric.shape == (2,)
    with pytest.raises(AttributeError):
        _ = decomposition.total
    with pytest.raises(KeyError):
        _ = decomposition["tu"]
    with pytest.raises(AttributeError):
        _ = decomposition.epistemic
    with pytest.raises(KeyError):
        _ = decomposition["eu"]


def test_measure_hetnets_sample_returns_aleatoric_uncertainty() -> None:
    sample = _hetnets_sample()

    uncertainty = measure(sample)
    decomposition = decompose(sample)

    assert torch.allclose(uncertainty, decomposition.aleatoric)
    assert uncertainty.shape == (2,)
