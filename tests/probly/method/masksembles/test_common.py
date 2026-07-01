"""Backend-agnostic tests for masksembles."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation.masksembles import masksembles


def test_raises_for_invalid_n_masks(dummy_predictor: Predictor) -> None:
    """Tests the behavior of the masksembles function when provided with an invalid number of masks.

    This function validates that the masksembles function raises a ValueError when
    the number of masks is not positive.

    Raises:
        ValueError: if the number of masks is nor positive.
    """
    n_masks = 0
    with pytest.raises(ValueError, match="n_masks"):
        masksembles(dummy_predictor, n_masks=n_masks, scale=2.0)


def test_raises_for_invalid_scale(dummy_predictor: Predictor) -> None:
    """Tests the behavior of the masksembles function when provided with an invalid scale.

    This function validates that the masksembles function raises a ValueError when
    the scale is not positive.

    Raises:
        ValueError: if scale is not positive.
    """
    scale = 0.0
    with pytest.raises(ValueError, match="scale"):
        masksembles(dummy_predictor, n_masks=4, scale=scale)
