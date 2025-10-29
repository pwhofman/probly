"""Test for classification models."""

from __future__ import annotations


import pytest

from probly.predictor import Predictor
from probly.transformation.evidential.classification import evidential_classification 

def test_invalid_input_in_evidential_classification(dummy_predictor: Predictor) -> None:

    """Tests the behavior of the evidential classification function when provided with a model that has no registered appender.

    """
    with pytest.raises(NotImplementedError, match=f"No evidential classification appender registered for type {type(dummy_predictor)}"):
        evidential_classification(dummy_predictor)
