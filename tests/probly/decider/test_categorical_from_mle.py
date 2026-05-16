"""Tests for the categorical_from_mle decider."""

from __future__ import annotations

import numpy as np
import pytest

from probly.decider import categorical_from_mean, categorical_from_mle
from probly.representation.credal_set.array import (
    ArrayMLEProbabilityIntervalsCredalSet,
    ArrayProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)


def _mle_credal_set() -> ArrayMLEProbabilityIntervalsCredalSet:
    return ArrayMLEProbabilityIntervalsCredalSet(
        lower_bounds=np.array([[0.1, 0.2, 0.3]]),
        upper_bounds=np.array([[0.3, 0.4, 0.5]]),
        mle=ArrayProbabilityCategoricalDistribution(np.array([[0.2, 0.3, 0.5]])),
    )


def test_categorical_from_mle_returns_stored_mle() -> None:
    credal = _mle_credal_set()

    single = categorical_from_mle(credal)

    assert isinstance(single, ArrayCategoricalDistribution)
    np.testing.assert_allclose(single.probabilities, [[0.2, 0.3, 0.5]])


def test_categorical_from_mle_raises_on_plain_probability_intervals() -> None:
    plain = ArrayProbabilityIntervalsCredalSet(
        lower_bounds=np.array([[0.1, 0.2, 0.3]]),
        upper_bounds=np.array([[0.3, 0.4, 0.5]]),
    )

    with pytest.raises(NotImplementedError, match="categorical_from_mle"):
        categorical_from_mle(plain)


def test_categorical_from_mean_on_mle_credal_set_returns_barycenter_not_mle() -> None:
    # Regression: keep mean and MLE semantically distinct.
    credal = _mle_credal_set()

    via_mean = categorical_from_mean(credal)
    via_mle = categorical_from_mle(credal)

    np.testing.assert_allclose(via_mean.probabilities, credal.barycenter.probabilities)
    # MLE differs from the geometric interval center for this construction.
    assert not np.allclose(via_mean.probabilities, via_mle.probabilities)
