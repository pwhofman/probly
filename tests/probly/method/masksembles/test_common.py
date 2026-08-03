"""Backend-agnostic tests for masksembles."""

from __future__ import annotations

import pytest

from probly.method.masksembles import masksembles
from probly.predictor import Predictor


class TestInvalidArgs:
    """Tests for invalid factory arguments."""

    def test_num_masks_must_be_positive(self, dummy_predictor: Predictor) -> None:
        num_masks = 0
        msg = f"num_masks must be a positive integer, got {num_masks}."
        with pytest.raises(ValueError, match=msg):
            masksembles(dummy_predictor, num_masks=num_masks)

    def test_scale_must_be_positive(self, dummy_predictor: Predictor) -> None:
        scale = -1.0
        msg = f"scale must be greater than 0 and at most 6.0, got {scale}."
        with pytest.raises(ValueError, match=msg):
            masksembles(dummy_predictor, scale=scale)

    def test_scale_must_be_at_most_six(self, dummy_predictor: Predictor) -> None:
        scale = 7.0
        msg = f"scale must be greater than 0 and at most 6.0, got {scale}."
        with pytest.raises(ValueError, match=msg):
            masksembles(dummy_predictor, scale=scale)
