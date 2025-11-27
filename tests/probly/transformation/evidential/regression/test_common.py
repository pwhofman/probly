"""Tests for evidential regression models."""

from __future__ import annotations

from typing import Any, cast

from probly.predictor import Predictor
from probly.transformation.evidential.regression import evidential_regression


def test_unknown_base_returns_self(dummy_predictor: Predictor) -> None:
    """Tests that the base model is returned if no implementation is registered.

    This uses the provided dummy_predictor fixture instead of a locally defined class.

    Parameters:
        dummy_predictor (Predictor): The generic predictor fixture supplied by pytest.
    """
    # Use the official fixture which adheres to the Predictor type.
    base = dummy_predictor

    transformed = evidential_regression(cast(Any, base))

    # Assert that the function returned the exact same object instance.
    assert transformed is base
