"""Backend-agnostic tests for the credal relative likelihood method."""

from __future__ import annotations

import pytest

from probly.method.credal_relative_likelihood import (
    relative_likelihood_thresholds,
    train_credal_relative_likelihood,
)


class TestRelativeLikelihoodThresholds:
    """Uniform interpolation over [alpha, 1) for the non-reference members."""

    def test_matches_benchmark_schedule(self) -> None:
        assert relative_likelihood_thresholds(0.5, 4) == pytest.approx([0.5, 0.5 + 0.5 / 3, 0.5 + 1.0 / 3])

    def test_first_target_is_alpha_and_last_below_one(self) -> None:
        thresholds = relative_likelihood_thresholds(0.8, 10)
        assert thresholds[0] == 0.8
        assert len(thresholds) == 9
        assert max(thresholds) < 1.0

    def test_single_member_has_no_targets(self) -> None:
        assert relative_likelihood_thresholds(0.5, 1) == []

    def test_alpha_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            relative_likelihood_thresholds(0.0, 4)
        with pytest.raises(ValueError, match="alpha"):
            relative_likelihood_thresholds(1.5, 4)

    def test_num_members_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="num_members"):
            relative_likelihood_thresholds(0.5, 0)


class TestTrainCredalRelativeLikelihoodDispatch:
    """Backend dispatch of the training facade."""

    def test_unregistered_predictor_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="No credal relative likelihood trainer"):
            train_credal_relative_likelihood(object(), None)
