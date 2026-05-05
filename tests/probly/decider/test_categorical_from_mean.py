"""Tests for categorical mean deciders."""

from __future__ import annotations

import numpy as np
import pytest

from probly.decider import categorical_from_mean, mean_field_categorical
from probly.representation.conformal_set.array import ArrayOneHotConformalSet
from probly.representation.credal_set.array import (
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
    ArrayProbabilityCategoricalDistribution,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution


def test_categorical_from_mean_returns_categorical_distribution_unchanged() -> None:
    distribution = ArrayProbabilityCategoricalDistribution(np.array([[0.2, 0.3, 0.5]]))

    assert categorical_from_mean(distribution) is distribution


def test_categorical_from_mean_reduces_categorical_sample_to_mean_distribution() -> None:
    sample = ArrayCategoricalDistributionSample(
        array=ArrayProbabilityCategoricalDistribution(
            np.array(
                [
                    [[2.0, 2.0, 0.0], [1.0, 3.0, 0.0]],
                    [[1.0, 1.0, 2.0], [3.0, 1.0, 0.0]],
                ]
            )
        ),
        sample_axis=0,
    )

    single = categorical_from_mean(sample)

    assert isinstance(single, ArrayCategoricalDistribution)
    np.testing.assert_allclose(single.probabilities, np.array([[0.375, 0.375, 0.25], [0.5, 0.5, 0.0]]))


def test_categorical_from_mean_reduces_dirichlet_to_expected_categorical_distribution() -> None:
    distribution = ArrayDirichletDistribution(np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0]]))

    single = categorical_from_mean(distribution)

    assert isinstance(single, ArrayCategoricalDistribution)
    np.testing.assert_allclose(single.probabilities, np.array([[1 / 6, 2 / 6, 3 / 6], [2 / 8, 2 / 8, 4 / 8]]))


def test_categorical_from_mean_reduces_probability_intervals_to_center_distribution() -> None:
    credal_set = ArrayProbabilityIntervalsCredalSet(
        lower_bounds=np.array([[0.1, 0.2, 0.3]]),
        upper_bounds=np.array([[0.3, 0.4, 0.5]]),
    )

    single = categorical_from_mean(credal_set)

    assert isinstance(single, ArrayCategoricalDistribution)
    np.testing.assert_allclose(single.probabilities, np.array([[7 / 30, 1 / 3, 13 / 30]]))


def test_categorical_from_mean_reduces_distance_based_credal_set_to_nominal_distribution() -> None:
    nominal = ArrayProbabilityCategoricalDistribution(np.array([[0.2, 0.3, 0.5]]))
    credal_set = ArrayDistanceBasedCredalSet(nominal=nominal, radius=np.array([0.1]))

    assert categorical_from_mean(credal_set) is nominal


def test_categorical_from_mean_reduces_one_hot_conformal_set_to_dense() -> None:
    conformal_set = ArrayOneHotConformalSet(
        np.array(
            [
                [True, False, False],
                [True, False, True],
            ]
        )
    )

    single = categorical_from_mean(conformal_set)

    assert isinstance(single, ArrayCategoricalDistribution)
    np.testing.assert_allclose(single.probabilities, np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.5]]))


torch_module = pytest.importorskip("torch")

from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution  # noqa: E402


def test_categorical_from_mean_dispatches_gaussian_to_mean_field_with_default_factor() -> None:
    gaussian = TorchGaussianDistribution(
        mean=torch_module.tensor([[2.0, -1.0, 0.5]], dtype=torch_module.float32),
        var=torch_module.tensor([[0.2, 0.4, 0.6]], dtype=torch_module.float32),
    )

    via_default = categorical_from_mean(gaussian)
    via_mean_field = mean_field_categorical(gaussian, mean_field_factor=1.0)

    assert torch_module.allclose(via_default.probabilities, via_mean_field.probabilities, atol=1e-6)
