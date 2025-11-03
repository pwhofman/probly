"""Test for evidential regression models."""

from __future__ import annotations

from typing import Any, cast

from probly.transformation.evidential.regression import evidential_regression


def test_unknown_base_returns_self() -> None:
    """Tests that the transformation returns the base model unchanged.

    if no implementation  (like Torch or Flax) is registered for its type.
    """

    class DummyPredictor:
        pass

    base = DummyPredictor()

    transformed = evidential_regression(cast(Any, base))

    assert transformed is base
