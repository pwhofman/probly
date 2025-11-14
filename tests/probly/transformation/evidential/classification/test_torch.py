"""Tests for the Torch evidential classification integration."""

from __future__ import annotations

from typing import Any

from torch import nn

from probly.transformation.evidential.classification import torch as torch_impl
from probly.transformation.evidential.classification.common import (
    evidential_classification_appender,
)


def test_append_activation_torch_returns_sequential() -> None:
    """Ensure append_activation_torch wraps a model in Sequential + Softplus."""
    # Dummy-Modell: einfacher Linear-Layer
    model = nn.Linear(4, 2)
    result = torch_impl.append_activation_torch(model)

    # Der Rückgabewert sollte nn.Sequential sein
    assert isinstance(result, nn.Sequential), "Result must be nn.Sequential"
    # Es sollten genau zwei Module enthalten sein
    assert len(result) == 2, f"Expected 2 layers, got {len(result)}"
    # Erstes Modul: dein Linear-Modell
    assert isinstance(result[0], nn.Linear)
    # Zweites Modul: Softplus-Aktivierung
    assert isinstance(result[1], nn.Softplus)


def test_torch_appender_is_registered() -> None:
    """Ensure the torch appender is correctly registered via common.register."""
    model = nn.Linear(3, 3)

    # Sollte automatisch den Torch-Appender verwenden
    result: Any = evidential_classification_appender(model)

    # Der Rückgabewert sollte ein Sequential mit Softplus sein
    assert isinstance(result, nn.Sequential)
    assert isinstance(result[1], nn.Softplus)
