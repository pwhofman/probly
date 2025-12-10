from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation.subensemble import subensemble
from probly.transformation.subensemble.common import subensemble_generator


class InvalidPredictor(Predictor):
    def __call__(self, x: int) -> int:
        return x


class ValidPredictor(Predictor):
    pass


def test_invalid_type(dummy_predictor: InvalidPredictor) -> None:
    """Test that an invalid type raises NotImplementedError."""
    msg = f"No subensemble generator is registered for type {type(dummy_predictor)}"
    with pytest.raises(NotImplementedError, match=msg):
        subensemble_generator(dummy_predictor)


def test_invalid_head_layer(dummy_predictor: ValidPredictor) -> None:
    """Test if head_layer has a valid value."""
    head_layer = 0
    msg = f"head_layer must be a positive number, but got head_layer={head_layer} instead."
    with pytest.raises(ValueError, match=msg):
        subensemble(base=dummy_predictor, num_heads=1, head_layer=head_layer)


def test_invalid_num_heads(dummy_predictor: ValidPredictor) -> None:
    """Test if num_heads has a valid value."""
    num_heads = -1
    msg = f"num_heads must be positive number, but got num_heads={num_heads} instead."
    with pytest.raises(ValueError, match=msg):
        subensemble(base=dummy_predictor, num_heads=num_heads)
