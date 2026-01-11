"""Test for batchensemble models."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation.batchensemble import batchensemble


class ValidPredictor(Predictor):
    pass


class TestInvalidArgs:
    """Tests for invalid args."""

    def test_num_members(self, dummy_predictor: ValidPredictor) -> None:
        num_members = -1
        msg = f"num_members must be a positive integer, got {num_members}."
        with pytest.raises(ValueError, match=msg):
            batchensemble(dummy_predictor, num_members=num_members)

    def test_s_std(self, dummy_predictor: ValidPredictor) -> None:
        num_members = 2
        s_std = -0.1
        msg = (
            f"The initial standard deviation of the input modulation s must be greater than 0, but got {s_std} instead."
        )
        with pytest.raises(ValueError, match=msg):
            batchensemble(dummy_predictor, num_members=num_members, s_std=s_std)

    def test_s_mean(self, dummy_predictor: ValidPredictor) -> None:
        num_members = 2
        s_mean = -0.1
        msg = f"The initial mean of the input modulation s must be greater than 0, but got {s_mean} instead."
        with pytest.raises(ValueError, match=msg):
            batchensemble(dummy_predictor, num_members=num_members, s_mean=s_mean)

    def test_r_std(self, dummy_predictor: ValidPredictor) -> None:
        num_members = 2
        r_std = -0.1
        msg = (
            "The initial standard deviation of the output modulation r must be greater than 0, "
            f"but got {r_std} instead."
        )
        with pytest.raises(ValueError, match=msg):
            batchensemble(dummy_predictor, num_members=num_members, r_std=r_std)

    def test_r_mean(self, dummy_predictor: ValidPredictor) -> None:
        num_members = 2
        r_mean = -0.1
        msg = f"The initial mean of the output modulation r must be greater than 0, but got {r_mean} instead."
        with pytest.raises(ValueError, match=msg):
            batchensemble(dummy_predictor, num_members=num_members, r_mean=r_mean)


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
