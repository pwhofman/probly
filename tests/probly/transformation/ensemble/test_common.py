from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation.ensemble.common import ensemble_generator


class InvalidPredictor(Predictor):
    def __call__(self, x: int) -> int:
        return x


class ValidPredictor(Predictor):
    pass


def test_invalid_type() -> None:
    """Test that an invalid type raises NotImplementedError."""
    n_members = 3
    base = InvalidPredictor()

    with pytest.raises(NotImplementedError):
        ensemble_generator(base, num_members=n_members)


def test_invalid_members() -> None:
    """Test n_members is a valid type."""
    n_members = 2.5

    with pytest.raises(AssertionError):
        assert isinstance(int, type(n_members))
