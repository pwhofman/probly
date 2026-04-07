"""Tests for Numpy-based categorical distribution representation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.distribution import create_categorical_distribution
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution
from probly.representation.sample import ArraySample


def test_accepts_relative_non_negative_probabilities() -> None:
    probabilities = np.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]], dtype=float)

    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    assert dist.shape == (2,)
    assert dist.num_classes == 3


def test_create_categorical_distribution_from_ndarray() -> None:
    probabilities = np.array([[2.0, 3.0, 5.0]], dtype=float)

    dist = create_categorical_distribution(probabilities)

    assert isinstance(dist, ArrayCategoricalDistribution)
    np.testing.assert_allclose(dist.probabilities, probabilities)


def test_create_categorical_distribution_from_sequence() -> None:
    dist = create_categorical_distribution([[0.1, 0.3, 0.6]])

    assert isinstance(dist, ArrayCategoricalDistribution)
    np.testing.assert_allclose(dist.probabilities, np.array([[0.1, 0.3, 0.6]], dtype=float))


def test_rejects_negative_relative_probabilities() -> None:
    probabilities = np.array([1.0, -1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="non-negative"):
        ArrayCategoricalDistribution(probabilities=probabilities)


def test_zero_sum_relative_probabilities_raise_when_normalization_needed() -> None:
    dist = ArrayCategoricalDistribution(probabilities=np.array([[0.0, 0.0, 0.0]], dtype=float))

    with pytest.raises(ValueError, match="strictly positive sum"):
        _ = dist.entropy

    with pytest.raises(ValueError, match="strictly positive sum"):
        dist.sample(10)


def test_bernoulli_validation_uses_unit_interval() -> None:
    ArrayCategoricalDistribution(probabilities=np.array([[0.0], [0.5], [1.0]], dtype=float))

    with pytest.raises(ValueError, match="Bernoulli probabilities"):
        ArrayCategoricalDistribution(probabilities=np.array([[1.1]], dtype=float))


def test_bernoulli_reports_two_classes() -> None:
    dist = ArrayCategoricalDistribution(probabilities=np.array([[0.2], [0.8]], dtype=float))

    assert dist.num_classes == 2


def test_entropy_normalizes_relative_probabilities() -> None:
    probabilities = np.array([[2.0, 3.0, 5.0]], dtype=float)
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    normalized = probabilities / probabilities.sum(axis=-1, keepdims=True)
    expected = -np.sum(normalized * np.log(normalized), axis=-1)

    np.testing.assert_allclose(dist.entropy, expected)


def test_entropy_bernoulli_formula() -> None:
    probabilities = np.array([[0.25], [0.5], [0.75]], dtype=float)
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    p = probabilities[:, 0]
    expected = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    np.testing.assert_allclose(dist.entropy, expected)


def test_sampling_relative_probabilities_matches_normalized_distribution() -> None:
    probabilities = np.array([[2.0, 3.0, 5.0]], dtype=float)
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    sample = dist.sample(num_samples=30_000, rng=np.random.default_rng(0))

    assert isinstance(sample, ArraySample)
    assert sample.sample_axis == 0
    assert sample.array.shape == (30_000, 1)
    assert sample.array.dtype == np.int64

    values, counts = np.unique(sample.array[:, 0], return_counts=True)
    frequencies = np.zeros(dist.num_classes)
    frequencies[values] = counts / counts.sum()
    expected = np.array([0.2, 0.3, 0.5], dtype=float)

    np.testing.assert_allclose(frequencies, expected, atol=0.02)


def test_sampling_bernoulli_produces_binary_samples_with_correct_mean() -> None:
    p = np.array([[0.3]], dtype=float)
    dist = ArrayCategoricalDistribution(probabilities=p)

    sample = dist.sample(num_samples=40_000, rng=np.random.default_rng(1))

    assert isinstance(sample, ArraySample)
    assert sample.array.shape == (40_000, 1)
    assert np.all((sample.array == 0) | (sample.array == 1))
    assert float(sample.array.mean()) == pytest.approx(0.3, abs=0.02)


def test_getitem_cannot_index_class_axis_directly() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    with pytest.raises(IndexError):
        _ = dist[:, :, 0]


def test_setitem_cannot_index_class_axis_directly() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    with pytest.raises(IndexError):
        dist[:, :, 0] = np.array([1.0, 2.0, 3.0, 4.0])


def test_expand_dims_last_inserts_before_class_axis() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    expanded = np.expand_dims(dist, axis=-1)

    assert isinstance(expanded, ArrayCategoricalDistribution)
    assert expanded.shape == (2, 3, 1)
    assert expanded.probabilities.shape == (2, 3, 1, 4)


def test_reshape_with_none_inserts_before_class_axis() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    reshaped = np.reshape(dist, (6, None))

    assert isinstance(reshaped, ArrayCategoricalDistribution)
    assert reshaped.shape == (6, 1)
    assert reshaped.probabilities.shape == (6, 1, 4)


def test_concatenate_preserves_distribution_type() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    concatenated = np.concatenate((dist, dist), axis=-1)

    assert isinstance(concatenated, ArrayCategoricalDistribution)
    assert concatenated.shape == (2, 6)
    assert concatenated.probabilities.shape == (2, 6, 4)


def test_concat_alias_preserves_distribution_type() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    concatenated = np.concat((dist, dist), axis=-1)

    assert isinstance(concatenated, ArrayCategoricalDistribution)
    assert concatenated.shape == (2, 6)
    assert concatenated.probabilities.shape == (2, 6, 4)


def test_stack_preserves_distribution_type() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayCategoricalDistribution(probabilities=probabilities)

    stacked = np.stack((dist, dist), axis=0)

    assert isinstance(stacked, ArrayCategoricalDistribution)
    assert stacked.shape == (2, 2, 3)
    assert stacked.probabilities.shape == (2, 2, 3, 4)
