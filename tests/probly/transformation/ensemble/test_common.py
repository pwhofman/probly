
"""Tests for the ensemble module."""
from __future__ import annotations

import pytest

from probly.predictor import Predictor
from tests.probly.fixtures.common import dummy_predictor
from probly.transformation import ensemble
from probly.transformation.ensemble.common import *
       
def test_base(dummy_predictor: Predictor) -> None:
    class DummyClass:
        pass
    def dummy_generator(base, n_members):
        return (base, n_members)
    register(cls = DummyClass, generator = dummy_generator)

def test_ensemble_generator_when_not_registered(dummy_predictor: Predictor) -> None:
    base = dummy_predictor
    class DummyClass:
        pass
    with pytest.raises(NotImplementedError, match="No ensemble generator is registered for type"):
        ensemble_generator(base)
        
def test_example_ensemble(dummy_predictor: Predictor) -> None:
    base = dummy_predictor
    k = 3
    def dummy_generator(base, n_members):
        return (base, n_members)
    register(cls=type(dummy_predictor), generator = dummy_generator)
    out = ensemble(base, n_members=k)
    assert out == (base, k)