"""Test for dropout models."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import dropout


def test_invalid_p_value(dummy_predictor: Predictor) -> None:
    """Tests the behavior of the dropout function when provided with an invalid probability value.

    This function validates that the dropout function raises a ValueError when
    the probability parameter `p` is outside the valid range [0, 1].

    Raises:
        ValueError: If the probability `p` is not between 0 and 1.
    """
    p = 2
    with pytest.raises(ValueError, match=f"The probability p must be between 0 and 1, but got {p} instead."):
        dropout(dummy_predictor, p=p)
