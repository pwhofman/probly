"""Backend-agnostic tests for the SWAG method."""

from __future__ import annotations

import re

import pytest

from probly.method.swag import swag, swag_from_snapshots
from probly.predictor import Predictor


def test_invalid_max_rank(dummy_predictor: Predictor) -> None:
    """Swag raises a ValueError for a negative maximum rank."""
    with pytest.raises(ValueError, match=re.escape("The maximum rank must be non-negative, but got -1 instead.")):
        swag(dummy_predictor, max_rank=-1)


def test_from_snapshots_invalid_max_rank(dummy_predictor: Predictor) -> None:
    """Swag_from_snapshots validates its arguments before dispatching."""
    with pytest.raises(ValueError, match=re.escape("The maximum rank must be non-negative, but got -1 instead.")):
        swag_from_snapshots(dummy_predictor, [], max_rank=-1)


def test_invalid_scale(dummy_predictor: Predictor) -> None:
    """Swag raises a ValueError for a negative scale."""
    with pytest.raises(ValueError, match=re.escape("The scale must be non-negative, but got -0.5 instead.")):
        swag(dummy_predictor, scale=-0.5)
