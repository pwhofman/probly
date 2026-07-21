"""Tests for the opt-in MLP factory."""

from __future__ import annotations

import torch
from torch import nn

from probly.predictor import LogitClassifier
from stacking.models import StackingMLP, build_mlp


def test_build_mlp_returns_module_with_correct_output_shape() -> None:
    """build_mlp returns a module that maps (B, in_features) -> (B, num_classes)."""
    model = build_mlp(in_features=8, num_classes=3, hidden=(16, 16))
    assert isinstance(model, nn.Module)
    x = torch.randn(5, 8)
    logits = model(x)
    assert logits.shape == (5, 3)


def test_build_mlp_satisfies_logit_classifier_protocol() -> None:
    """build_mlp instance is recognised by isinstance(LogitClassifier).

    This is what lets probly's calibration / conformal layers wrap it.
    """
    model = build_mlp(in_features=4, num_classes=2)
    assert isinstance(model, LogitClassifier)


def test_stacking_mlp_class_directly_instantiable() -> None:
    """The StackingMLP class is importable and instantiable directly."""
    model = StackingMLP(in_features=2, num_classes=2, hidden=(8,))
    x = torch.randn(3, 2)
    logits = model(x)
    assert logits.shape == (3, 2)
    assert isinstance(model, LogitClassifier)
