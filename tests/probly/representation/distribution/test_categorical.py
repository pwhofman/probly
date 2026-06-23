"""Tests for Numpy-based categorical distribution representation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.distribution import create_categorical_distribution
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)
from probly.representation.sample import ArraySample


def test_accepts_relative_non_negative_probabilities() -> None:
    probabilities = np.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]], dtype=float)

    dist = ArrayProbabilityCategoricalDistribution(probabilities)

    assert dist.shape == (2,)
    assert dist.num_classes == 3


def test_create_categorical_distribution_from_ndarray() -> None:
    probabilities = np.array([[2.0, 3.0, 5.0]], dtype=float)

    dist = create_categorical_distribution(probabilities)

    assert isinstance(dist, ArrayCategoricalDistribution)
    np.testing.assert_allclose(dist.unnormalized_probabilities, probabilities)


def test_create_categorical_distribution_from_sequence() -> None:
    dist = create_categorical_distribution([[0.1, 0.3, 0.6]])

    assert isinstance(dist, ArrayCategoricalDistribution)
    np.testing.assert_allclose(dist.probabilities, np.array([[0.1, 0.3, 0.6]], dtype=float))


def test_rejects_negative_relative_probabilities() -> None:
    probabilities = np.array([1.0, -1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="non-negative"):
        ArrayProbabilityCategoricalDistribution(probabilities)


def test_zero_sum_relative_probabilities_return_nan() -> None:
    dist = ArrayProbabilityCategoricalDistribution(np.array([0.0, 0.0, 0.0], dtype=float))

    assert np.isnan(dist.probabilities).all()


def test_entropy_normalizes_relative_probabilities() -> None:
    probabilities = np.array([[2.0, 3.0, 5.0]], dtype=float)
    dist = ArrayProbabilityCategoricalDistribution(probabilities)

    normalized = probabilities / probabilities.sum(axis=-1, keepdims=True)
    expected = -np.sum(normalized * np.log(normalized), axis=-1)

    np.testing.assert_allclose(dist.entropy(), expected)


def test_sampling_relative_probabilities_matches_normalized_distribution() -> None:
    probabilities = np.array([[2.0, 3.0, 5.0]], dtype=float)
    dist = ArrayProbabilityCategoricalDistribution(probabilities)

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


def test_getitem_cannot_index_class_axis_directly() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayProbabilityCategoricalDistribution(probabilities)

    with pytest.raises(IndexError):
        _ = dist[:, :, 0]


def test_setitem_cannot_index_class_axis_directly() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayProbabilityCategoricalDistribution(probabilities)

    with pytest.raises(IndexError):
        dist[:, :, 0] = np.array([1.0, 2.0, 3.0, 4.0])


def test_expand_dims_last_inserts_before_class_axis() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayProbabilityCategoricalDistribution(probabilities)

    expanded = np.expand_dims(dist, axis=-1)

    assert isinstance(expanded, ArrayCategoricalDistribution)
    assert expanded.shape == (2, 3, 1)
    assert expanded.probabilities.shape == (2, 3, 1, 4)


def test_reshape_with_none_inserts_before_class_axis() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayProbabilityCategoricalDistribution(probabilities)

    reshaped = dist.reshape((6, None))

    assert isinstance(reshaped, ArrayCategoricalDistribution)
    assert reshaped.shape == (6, 1)
    assert reshaped.probabilities.shape == (6, 1, 4)


def test_concatenate_preserves_distribution_type() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayProbabilityCategoricalDistribution(probabilities)

    concatenated = np.concatenate((dist, dist), axis=-1)

    assert isinstance(concatenated, ArrayCategoricalDistribution)
    assert concatenated.shape == (2, 6)
    assert concatenated.probabilities.shape == (2, 6, 4)


def test_concat_alias_preserves_distribution_type() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayProbabilityCategoricalDistribution(probabilities)

    concatenated = np.concat((dist, dist), axis=-1)

    assert isinstance(concatenated, ArrayCategoricalDistribution)
    assert concatenated.shape == (2, 6)
    assert concatenated.probabilities.shape == (2, 6, 4)


def test_stack_preserves_distribution_type() -> None:
    probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
    dist = ArrayProbabilityCategoricalDistribution(probabilities)

    stacked = np.stack((dist, dist), axis=0)

    assert isinstance(stacked, ArrayCategoricalDistribution)
    assert stacked.shape == (2, 2, 3)
    assert stacked.probabilities.shape == (2, 2, 3, 4)


def test_mean_preserves_distribution_type_and_class_axis() -> None:
    unnormalized = np.array(
        [
            [[1.0, 1.0], [1.0, 3.0]],
            [[9.0, 1.0], [2.0, 2.0]],
        ],
        dtype=float,
    )
    dist = ArrayProbabilityCategoricalDistribution(unnormalized)

    meaned = np.mean(dist, axis=0)

    assert isinstance(meaned, ArrayCategoricalDistribution)
    assert meaned.shape == (2,)
    expected = np.mean(unnormalized / np.sum(unnormalized, axis=-1, keepdims=True), axis=0)
    np.testing.assert_allclose(meaned.probabilities, expected)


def test_average_preserves_distribution_type_and_uses_weights() -> None:
    unnormalized = np.array(
        [
            [[1.0, 1.0], [1.0, 3.0]],
            [[9.0, 1.0], [2.0, 2.0]],
        ],
        dtype=float,
    )
    weights = np.array([0.25, 0.75])
    dist = ArrayProbabilityCategoricalDistribution(unnormalized)

    averaged = np.average(dist, axis=0, weights=weights)

    assert isinstance(averaged, ArrayCategoricalDistribution)
    assert averaged.shape == (2,)
    probabilities = unnormalized / np.sum(unnormalized, axis=-1, keepdims=True)
    expected = np.average(probabilities, axis=0, weights=weights)
    np.testing.assert_allclose(averaged.probabilities, expected)


def test_hash_is_identity_based_and_distinguishes_instances() -> None:
    probabilities = np.array([[0.2, 0.8]], dtype=float)
    dist_a = ArrayProbabilityCategoricalDistribution(probabilities.copy())
    dist_b = ArrayProbabilityCategoricalDistribution(probabilities.copy())

    assert hash(dist_a) == hash(dist_a)
    assert hash(dist_a) != hash(dist_b)


class TestArrayCategoricalDistributionPostprocessing:
    """Ensure protected-axis processing rebuilds a ProbabilityCategoricalDistribution after np.mean."""

    def test_mean_returns_probability_distribution(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayLogitCategoricalDistribution,
            ArrayProbabilityCategoricalDistribution,
        )

        # A LogitCategorical reduced via np.mean should land as a Probability distribution.
        d = ArrayLogitCategoricalDistribution(array=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))
        result = np.mean(d, axis=0)
        # The mean of a logit-cat distribution should produce a probability distribution.
        assert isinstance(result, ArrayProbabilityCategoricalDistribution)


class TestArrayCategoricalDistribution:
    """Validation, equality and sampling for the numpy categorical distribution."""

    def test_negative_probabilities_raise(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="non-negative"):
            ArrayProbabilityCategoricalDistribution(array=np.array([0.5, -0.1, 0.6]))

    def test_zero_dim_array_raises(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="at least one dimension"):
            ArrayProbabilityCategoricalDistribution(array=np.array(0.5))

    def test_array_must_be_ndarray(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        with pytest.raises(TypeError, match="numpy ndarray"):
            ArrayProbabilityCategoricalDistribution(array=[0.5, 0.5])  # type: ignore[arg-type]

    def test_logit_array_must_be_ndarray(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayLogitCategoricalDistribution,
        )

        with pytest.raises(TypeError, match="numpy ndarray"):
            ArrayLogitCategoricalDistribution(array=[0.5, 0.5])  # type: ignore[arg-type]

    def test_logit_zero_dim_raises(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayLogitCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="at least one dimension"):
            ArrayLogitCategoricalDistribution(array=np.array(0.5))

    def test_eq_two_probability_distributions(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        d1 = ArrayProbabilityCategoricalDistribution(array=np.array([[0.2, 0.3, 0.5]]))
        d2 = ArrayProbabilityCategoricalDistribution(array=np.array([[0.4, 0.6, 1.0]]))
        # After normalization both have the same probabilities.
        assert bool(d1 == d2)

    def test_eq_with_array(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        d1 = ArrayProbabilityCategoricalDistribution(array=np.array([[0.2, 0.3, 0.5]]))
        # Comparison with a raw array uses unnormalised probabilities.
        eq = d1 == np.array([[0.2, 0.3, 0.5]])
        assert bool(eq)

    def test_logit_eq(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayLogitCategoricalDistribution,
        )

        a = ArrayLogitCategoricalDistribution(array=np.array([[1.0, 2.0, 3.0]]))
        b = ArrayLogitCategoricalDistribution(array=np.array([[1.0, 2.0, 3.0]]))
        assert bool(a == b)

    def test_logit_eq_with_array(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayLogitCategoricalDistribution,
        )

        a = ArrayLogitCategoricalDistribution(array=np.array([[1.0, 2.0, 3.0]]))
        eq = a == np.array([[1.0, 2.0, 3.0]])
        assert bool(eq)

    def test_hash(self) -> None:
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayLogitCategoricalDistribution,
            ArrayProbabilityCategoricalDistribution,
        )

        a = ArrayProbabilityCategoricalDistribution(array=np.array([[0.5, 0.5]]))
        b = ArrayLogitCategoricalDistribution(array=np.array([[0.0, 1.0]]))
        # Identity-based hash returns ints.
        assert isinstance(hash(a), int)
        assert isinstance(hash(b), int)
