"""Tests for common dropconnect models."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import dropconnect


""" def test_invalid_p_value(dummy_predictor: Predictor) -> None:
    Tests the behavior of the dropconnect function when provided with an invalid probability value.

    This function validates that the dropconnect function raises a ValueError when
    the probability parameter `p` is outside the valid range [0, 1].

    Raises:
        ValueError: If the probability `p` is not between 0 and 1.
    
    p = 2
    with pytest.raises(ValueError, match=f"The probability p must be between 0 and 1, but got {p} instead."):
        dropconnect(dummy_predictor, p=p)
        
            PROBLEM: common.py does not check the invalid_p_value"""


def test_valid_p_values(dummy_predictor: Predictor) -> None:
    """Tests that dropconnect works with valid p values"""
    
    model_0 = dropconnect(dummy_predictor, p=0.0)
    model_1 = dropconnect(dummy_predictor, p=1.0)
    model_half = dropconnect(dummy_predictor, p=0.5)
    model_invalid = dropconnect(dummy_predictor, p=2)
    
    assert model_0 is not None
    assert model_1 is not None
    assert model_half is not None
    assert model_invalid is not None