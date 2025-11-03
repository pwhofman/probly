"""Common tests for dropconnect models."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import dropconnect


def test_invalid_p_value(dummy_predictor: Predictor) -> None:
    """Tests the behavior of the dropconnect function when provided with an invalid probability value.

    This function validates that the dropconnect function raises a ValueError when
    the probability parameter `p` is outside the valid range [0, 1].

    Raises:
        ValueError: If the probability `p` is not between 0 and 1.
    """
    p = 2
    with pytest.raises(ValueError, match=f"The probability p must be between 0 and 1, but got {p} instead."):
        dropconnect(dummy_predictor, p=p)


def test_valid_p_values(dummy_predictor: Predictor) -> None:
    """Tests that dropconnect works with valid p values.
    
    This function tests that dropconnect accepts valid probability values
    in the range [0, 1] without raising errors.
    """
    # Test boundary values and typical values
    model_0 = dropconnect(dummy_predictor, p=0.0)    # No dropconnect
    model_1 = dropconnect(dummy_predictor, p=1.0)    # Full dropconnect  
    model_half = dropconnect(dummy_predictor, p=0.5) # 50% dropconnect
    
    # All should be created successfully without errors
    assert model_0 is not None
    assert model_1 is not None
    assert model_half is not None


