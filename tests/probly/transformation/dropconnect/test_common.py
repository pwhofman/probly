"""Tests for DropConnect models."""

from __future__ import annotations

from probly.predictor import Predictor
from probly.transformation import dropconnect


class DummyPredictor(Predictor):
    def initdummy(self, param: float = 1.0) -> None:
        self.param = param

    def predict(self, x: float) -> float:
        return x * self.param


def test_valid_value_creates_valid_copy(dummy_predictor: Predictor) -> None:
    p = 0.5
    model = dropconnect(dummy_predictor, p=p)
    assert isinstance(model, type(dummy_predictor))
    assert model is not dummy_predictor


def test_dropconnect_preserves_predictor_type() -> None:
    model = DummyPredictor()
    dropconnect_model = dropconnect(model, p=0.5)
    assert isinstance(dropconnect_model, DummyPredictor)
    assert dropconnect_model.param == model.param
