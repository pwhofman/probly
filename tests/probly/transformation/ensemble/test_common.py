"""Tests for ensemble models."""

from __future__ import annotations
import pytest

from probly.predictor import Predictor
from probly.transformation.ensemble import ensemble
from probly.transformation.ensemble.common import register

# def test_invalid_members(dummy_predictor: Predictor) -> None: 
#     """test if num_members < 1"""

#     register(Predictor, lambda base, num_members, reset_params=True: base)

#     with pytest.raises(ValueError, match="num_members must be >= 1"):
#         ensemble(dummy_predictor, num_members=0)

def test_ensemble_return_object(dummy_predictor: Predictor) -> None: 
    """test if ensemble returns an object with a predict method"""

    def simple_generator(base: Predictor, num_members: int, reset_params: bool = True): 
        class Wrapper: 
            def predict(self, x): 
                return base.predict(x)
        return Wrapper()
    
    register(Predictor, simple_generator)

    en = ensemble(dummy_predictor, num_members=3)
    
    assert en is not None
    assert hasattr(en, "predict")

def test_ensemble_correct_average(dummy_predictor: Predictor) -> None: 
    """test for the correct output by ensemble"""

    def simple_generator(base: Predictor, num_members: int, reset_params: bool = True): 
        class Wrapper: 
            def __init__(self, members): 
                self._members = members

            def predict(self, inputs): 
                outputs = [m.predict(inputs) for m in self._members]

                return [sum(vals)/len(vals) for vals in zip(*outputs)]
            
        members = [base for _ in range(num_members)]
        return Wrapper(members)
    
    register(Predictor, simple_generator)

    en = ensemble(dummy_predictor, num_members=5)

    inputs = [0, 1, 2]
    out = en.predict(inputs)

    expected = dummy_predictor.predict(inputs)
    assert out == expected
