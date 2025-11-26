"""Tests for ensemble models."""

from __future__ import annotations

from probly.predictor import Predictor, predict
from probly.transformation.ensemble import ensemble
from probly.transformation.ensemble.common import register


def test_ensemble_return_object(dummy_predictor: Predictor) -> None:
    """Test if ensemble returns an object with a predict method."""

    def simple_generator(base: Predictor, num_members: int, reset_params: bool = True) -> object: # noqa: ARG001
        class Wrapper:
            def predict(self, x: object) -> object:
                return predict(base, x)
        return Wrapper()

    register(Predictor, simple_generator)

    en = ensemble(dummy_predictor, num_members=3)

    assert en is not None
    assert hasattr(en, "predict")

def test_ensemble_correct_average(dummy_predictor: Predictor) -> None:
    """Test for the correct output by ensemble."""

    def simple_generator(base: Predictor, num_members: int, reset_params: bool = True): # noqa: ARG001
        class Wrapper:
            def __init__(self, members) -> None:
                self._members = members

            def predict(self, inputs):
                outputs = [predict(m, inputs) for m in self._members]

                return [sum(vals)/len(vals) for vals in zip(*outputs, strict=False)]

        members = [base for _ in range(num_members)]
        return Wrapper(members)

    register(Predictor, simple_generator)

    en = ensemble(dummy_predictor, num_members=5)

    inputs = [0, 1, 2]
    out = en.predict(inputs)

    expected = predict(dummy_predictor, inputs)
    assert out == expected
