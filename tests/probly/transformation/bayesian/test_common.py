from __future__ import annotations

import pytest

from probly.predictor import Predictor 
from probly.transformation import bayesian

def test_bayesian(example_predictor: Predictor) -> None:        # Test Bayesian transformation on a generic predictor.
    bayesian_predictor = bayesian(
        example_predictor,
        posterior=0.1,
        prior_mean=0.0,
        prior=1.0,
        )                                                       # Create Bayesian predictor.
    
    assert isinstance(bayesian_predictor, Predictor)            # Check if the result is still a Predictor.
        
    assert hasattr(bayesian_predictor, "predict")               # Ensure the predictor has a predict method.
    assert bayesian_predictor.model is not None                 # Ensure the model attribute is present.    

 
def test_invalid_parameters(example_predictor: Predictor) -> None:  # Test Bayesian transformation with invalid parameters.
    with pytest.raises(ValueError):
        bayesian(
            example_predictor,
            posterior=-0.1,             # Invalid negative posterior std (<=0)
            prior_mean=0.0,
            prior=1.0,
        )
    
    with pytest.raises(ValueError):
        bayesian(
            example_predictor,
            posterior=0.1,
            prior_mean=0.0,
            prior=-1.0,                 # Invalid negative prior std (<=0)
        )