
"""Tests for the ensemble module."""
from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import ensemble
       
def test_ensemble_generator_when_not_registered(dummy_predictor) -> None:
    n_members = 3
    with pytest.raises(NotImplementedError, match=f"No ensemble generator is registered for type",):
        ensemble(dummy_predictor, n_members=n_members)