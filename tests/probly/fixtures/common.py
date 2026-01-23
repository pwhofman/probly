"""Common fixtures for tests."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor


@pytest.fixture
def dummy_predictor() -> Predictor:
    """Return a dummy predictor."""

    class DummyPredictor(Predictor):
        def __call__(self, x: float) -> float:
            return x

        def predict(self, x: float) -> float:
            return self.__call__(x)

    return DummyPredictor()
