"""Test for ensemble models."""
from __future__ import annotations
import pytest
from probly.predictor import Predictor
from probly.transformation import ensemble
from probly.transformation.ensemble.common import *
from tests.probly.fixtures.common import dummy_predictor


def test_invalid_base_module(dummy_predictor: Predictor):
    """Base module without generator should throw an NotImplementedError"""
    base = dummy_predictor
    with pytest.raises(NotImplementedError):
        ensemble_generator(base)

def test_invalid_register(dummy_predictor: Predictor) -> None:
    """Test that register function works."""

    class DummyClass:
        pass
    def dummy_gen(base, n_members):
        return (base, n_members)


    register(cls=DummyClass,generator=dummy_gen)


def test_ensemble_no_error(dummy_predictor: Predictor) -> None:
    """Test that ensemble function runs without error."""

    # Reg base first to use in ensemble
    def dummy_gen(base, n_members):
        return base

    register(cls=type(dummy_predictor), generator=dummy_gen)


    ensemble(base=dummy_predictor, n_members=5)
