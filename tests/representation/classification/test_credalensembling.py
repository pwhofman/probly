"""Tests for the credalensembling module."""

from __future__ import annotations

import pytest
import torch

from probly.representation.classification.credalensembling import CredalEnsembling


@pytest.fixture
def ensemble(conv_linear_model: torch.nn.Module) -> CredalEnsembling:
    return CredalEnsembling(conv_linear_model, n_members=5)


def test_predict_representation(ensemble: CredalEnsembling) -> None:
    inputs = torch.randn(2, 3, 5, 5)
    outputs = ensemble.predict_representation(inputs, alpha=0.0)
    assert outputs.shape == (2, 5, 2)
    outputs = ensemble.predict_representation(inputs, alpha=0.5)
    assert outputs.shape == (2, 2, 2)
    with pytest.raises(ValueError, match="Unknown distance metric: unknown"):
        ensemble.predict_representation(inputs, alpha=0.5, distance="unknown")
