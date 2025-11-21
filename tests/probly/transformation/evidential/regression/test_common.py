from __future__ import annotations


import pytest

from probly.predictor import Predictor
from probly.transformation.evidential.regression.common import evidential_regression
from probly.transformation.evidential.regression.common import register

def test_predict_method(dummy_predictor: Predictor) -> None: 
    """test if evidential_regression returns an object with a predict method"""

    def simple_generator(base: Predictor): 
        class Wrapper: 
            def predict(self, x): 
                return base.predict(x)
        return Wrapper()
    
    register(Predictor, simple_generator)

    model = evidential_regression(dummy_predictor)
    
    assert model is not None
    assert hasattr(model, "predict")