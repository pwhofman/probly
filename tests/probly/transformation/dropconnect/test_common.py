"""Test for dropconnect models."""

from __future__ import annotations

from probly.predictor import Predictor
from probly.transformation import dropconnect


def test_function_exists(dummy_predictor: Predictor) -> None:
    """Tests that dropconnect function exists and returns a Predictor."""
    model = dropconnect(dummy_predictor, p=0.25)
    assert isinstance(model, Predictor)


def test_clones_model(dummy_predictor: Predictor) -> None:
    """Tests that dropconnect function returns the cloned model, not the original."""
    model = dropconnect(dummy_predictor, p=0.25)
    assert model is not dummy_predictor
