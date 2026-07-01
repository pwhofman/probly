"""Backend-agnostic tests for masksembles."""

from __future__ import annotations

import pytest
from torch import nn

from probly.transformation.masksembles import masksembles


class TestMasksemblesFactory:
    """Tests for the masksembles() factory."""

    def test_raises_for_invalid_n_masks(self) -> None:

        with pytest.raises(ValueError, match="n_masks"):
            masksembles(nn.Linear(20, 10), n_masks=0, predictor_type="logit_classifier")

    def test_raises_for_invalid_scale(self) -> None:

        with pytest.raises(ValueError, match="scale"):
            masksembles(nn.Linear(20, 10), n_masks=4, scale=0.0, predictor_type="logit_classifier")
