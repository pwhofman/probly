"""Test for batchensemble models."""

from __future__ import annotations

import pytest

from probly.method.batchensemble import batchensemble
from probly.predictor import Predictor


class ValidPredictor(Predictor):
    pass


class TestInvalidArgs:
    """Tests for invalid args."""

    def test_num_members(self, dummy_predictor: ValidPredictor) -> None:
        num_members = -1
        msg = f"num_members must be a positive integer, got {num_members}."
        with pytest.raises(ValueError, match=msg):
            batchensemble(dummy_predictor, num_members=num_members)

    def test_invalid_init(self, dummy_predictor: ValidPredictor) -> None:
        msg = "init must be 'normal' or 'random_sign', got 'gaussian'."
        with pytest.raises(ValueError, match=msg):
            batchensemble(dummy_predictor, num_members=2, init="gaussian")

    def test_normal_r_std_must_be_positive(self, dummy_predictor: ValidPredictor) -> None:
        r_std = -0.1
        msg = f"r_std must be greater than 0 when init='normal', got {r_std}."
        with pytest.raises(ValueError, match=msg):
            batchensemble(dummy_predictor, num_members=2, init="normal", r_std=r_std)

    def test_normal_s_std_must_be_positive(self, dummy_predictor: ValidPredictor) -> None:
        s_std = -0.1
        msg = f"s_std must be greater than 0 when init='normal', got {s_std}."
        with pytest.raises(ValueError, match=msg):
            batchensemble(dummy_predictor, num_members=2, init="normal", s_std=s_std)


class InvalidPredictor(Predictor):
    def __call__(self, x: int) -> int:
        return x


@pytest.mark.skip(reason="not implemented yet")
class TestInvalidPredictor:
    """Tests for invalid predictor."""

    def test_invalid_predictor(self, dummy_predictor: InvalidPredictor) -> None:
        """Test that an invalid type raises NotImplementedError."""
        num_members = 2
        msg = f"No ensemble generator is registered for type {type(dummy_predictor)}"
        with pytest.raises(NotImplementedError, match=msg):
            batchensemble(dummy_predictor, num_members=num_members)
