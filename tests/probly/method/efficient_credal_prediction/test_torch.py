"""Tests for the torch backend of efficient credal prediction.

These tests guard against a subtle Protocol-default-implementation gap:
:class:`probly.method.efficient_credal_prediction.EfficientCredalPredictor`
declares ``lower_bounds`` / ``upper_bounds`` as ``@property`` defaults that
return ``self.lower`` / ``self.upper``. Python ``Protocol`` defaults do **not**
transfer to classes that satisfy the Protocol structurally (via
``runtime_checkable``) without explicit inheritance, so the concrete
:class:`~probly.method.efficient_credal_prediction.torch.TorchEfficientCredalPredictor`
must implement these properties itself. Without them, ``predict()`` crashes
with ``AttributeError`` because its dispatch reads
``predictor.lower_bounds`` / ``predictor.upper_bounds`` directly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
from torch import nn

from probly.method.efficient_credal_prediction import efficient_credal_prediction
from probly.predictor import predict
from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet


class _TinyClassifier(nn.Module):
    """Minimal logit classifier for testing."""

    def __init__(self, in_features: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_predict_uses_lower_upper_bounds_properties() -> None:
    """``predict()`` accesses ``lower_bounds`` / ``upper_bounds`` and must not crash.

    Regression test: see module docstring. Sets the buffers manually rather
    than running real bound estimation since this test only verifies the
    property-access path, not the bound-fitting algorithm.
    """
    base = _TinyClassifier(in_features=4, num_classes=3)
    cep = efficient_credal_prediction(base, predictor_type="logit_classifier")

    # Replace the None buffers with dummy bounds. Real training would compute
    # these via the credal-bounds optimization step.
    num_classes = 3
    cep.lower = torch.full((num_classes,), 0.05)
    cep.upper = torch.full((num_classes,), 0.05)

    # Property access path: this is the regression check.
    assert torch.equal(cep.lower_bounds, cep.lower)
    assert torch.equal(cep.upper_bounds, cep.upper)

    x = torch.zeros(2, 4)
    rep = predict(cep, x)
    assert isinstance(rep, TorchProbabilityIntervalsCredalSet)
    assert rep.lower_bounds.shape == (2, num_classes)
    assert rep.upper_bounds.shape == (2, num_classes)
