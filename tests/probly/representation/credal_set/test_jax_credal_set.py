"""Tests for jax-backed categorical credal sets."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from probly.representation.credal_set._common import (
    create_distance_based_credal_set_from_center_and_radius,
)
from probly.representation.credal_set.jax import (
    JaxConvexCredalSet,
    JaxDistanceBasedCredalSet,
    JaxProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.jax_functions import jax_expand_dims
from probly.representation.sample.jax import JaxArraySample


def test_jax_convex_credal_set_from_distribution_sample() -> None:
    probs = jnp.array(
        [
            [[0.1, 0.9], [0.4, 0.6]],
            [[0.2, 0.8], [0.5, 0.5]],
            [[0.15, 0.85], [0.45, 0.55]],
        ],
        dtype=float,
    )
    sample = JaxArraySample(
        array=JaxProbabilityCategoricalDistribution(probs),
        sample_axis=0,
    )

    cset = JaxConvexCredalSet.from_jax_sample(sample)

    assert isinstance(cset.tensor, JaxCategoricalDistribution)
    assert tuple(cset.tensor.probabilities.shape) == (2, 3, 2)


def test_jax_convex_credal_set_barycenter_averages_normalized_probabilities() -> None:
    vertices = jnp.array([[1.0, 1.0], [9.0, 1.0]], dtype=float)
    cset = JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(vertices))

    barycenter = cset.barycenter

    assert isinstance(barycenter, JaxCategoricalDistribution)
    assert jnp.allclose(barycenter.probabilities, jnp.array([0.7, 0.3], dtype=float))


def test_jax_probability_intervals_numpy_and_shape_ops() -> None:
    probs = jnp.array(
        [
            [[0.2, 0.8], [0.6, 0.4]],
            [[0.1, 0.9], [0.5, 0.5]],
        ],
        dtype=float,
    )
    sample = JaxArraySample(
        array=JaxProbabilityCategoricalDistribution(probs),
        sample_axis=0,
    )

    cset = JaxProbabilityIntervalsCredalSet.from_jax_sample(sample)
    arr = np.asarray(cset)

    assert arr.shape == (2, 2, 2)

    expanded = jax_expand_dims(cset, axis=0)
    assert isinstance(expanded, JaxProbabilityIntervalsCredalSet)
    assert tuple(expanded.lower_bounds.shape) == (1, 2, 2)
    assert tuple(expanded.upper_bounds.shape) == (1, 2, 2)


def test_distance_credal_set_from_categorical_distribution() -> None:
    """Factory should accept JaxCategoricalDistribution directly."""
    probs = jnp.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], dtype=float)
    center = JaxProbabilityCategoricalDistribution(probs)
    radius = jnp.array(0.1, dtype=float)

    result = create_distance_based_credal_set_from_center_and_radius(center, radius)

    assert isinstance(result, JaxDistanceBasedCredalSet)
    assert result.nominal is center
    expected_radius = jnp.broadcast_to(radius, (probs.shape[0],))
    assert jnp.array_equal(result.radius, expected_radius)


class TestJaxConvexCredalSet:
    """Convex credal sets backed by jax arrays."""

    def test_from_jax_sample(self) -> None:
        probs = jnp.array([[[0.5, 0.5]], [[0.3, 0.7]], [[0.6, 0.4]]])
        sample = JaxArraySample(
            array=JaxProbabilityCategoricalDistribution(probs),
            sample_axis=0,
        )
        cred = JaxConvexCredalSet.from_jax_sample(sample)
        assert cred.tensor.array.shape[-1] == 2

    def test_lower_upper(self) -> None:
        probs = jnp.array([[[0.5, 0.5], [0.3, 0.7]]])
        cred = JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(probs))
        assert jnp.allclose(cred.lower(), jnp.array([[0.3, 0.5]]))
        assert jnp.allclose(cred.upper(), jnp.array([[0.5, 0.7]]))

    def test_num_classes(self) -> None:
        probs = jnp.array([[[0.5, 0.5], [0.3, 0.7]]])
        cred = JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(probs))
        assert cred.num_classes == 2


class TestJaxDistanceBasedCredalSet:
    """Distance-based credal sets backed by jax arrays."""

    def test_from_jax_sample(self) -> None:
        probs = jnp.array([[[0.5, 0.5]], [[0.3, 0.7]]])
        sample = JaxArraySample(
            array=JaxProbabilityCategoricalDistribution(probs),
            sample_axis=0,
        )
        cred = JaxDistanceBasedCredalSet.from_jax_sample(sample)
        assert jnp.allclose(jnp.squeeze(cred.nominal.array), jnp.array([0.4, 0.6]), atol=1e-5, rtol=1e-5)
        assert cred.radius.shape == (1,)

    def test_lower_upper(self) -> None:
        cred = JaxDistanceBasedCredalSet(
            nominal=jnp.array([[0.4, 0.6]]),
            radius=jnp.array([0.1]),
        )
        assert jnp.allclose(cred.lower(), jnp.array([[0.3, 0.5]]))
        assert jnp.allclose(cred.upper(), jnp.array([[0.5, 0.7]]))

    def test_lower_clipped(self) -> None:
        cred = JaxDistanceBasedCredalSet(
            nominal=jnp.array([[0.05, 0.95]]),
            radius=jnp.array([0.5]),
        )
        assert jnp.allclose(cred.lower(), jnp.array([[0.0, 0.45]]))

    def test_upper_clipped(self) -> None:
        cred = JaxDistanceBasedCredalSet(
            nominal=jnp.array([[0.95, 0.05]]),
            radius=jnp.array([0.5]),
        )
        assert jnp.allclose(cred.upper(), jnp.array([[1.0, 0.55]]))


class TestJaxProbabilityIntervalsCredalSet:
    """Probability-interval credal sets backed by jax arrays."""

    def test_width(self) -> None:
        cred = JaxProbabilityIntervalsCredalSet(
            lower_bounds=jnp.array([[0.1, 0.2]]),
            upper_bounds=jnp.array([[0.5, 0.6]]),
        )
        assert jnp.allclose(cred.width(), jnp.array([[0.4, 0.4]]))

    def test_contains(self) -> None:
        cred = JaxProbabilityIntervalsCredalSet(
            lower_bounds=jnp.array([[0.1, 0.2]]),
            upper_bounds=jnp.array([[0.5, 0.6]]),
        )
        assert bool(cred.contains(jnp.array([[0.3, 0.4]])))
        assert not bool(cred.contains(jnp.array([[0.7, 0.5]])))

    def test_numpy_method(self) -> None:
        cred = JaxProbabilityIntervalsCredalSet(
            lower_bounds=jnp.array([[0.1, 0.2]]),
            upper_bounds=jnp.array([[0.5, 0.6]]),
        )
        arr = cred.numpy()
        assert arr.shape == (1, 2, 2)

    def test_numpy_method_force_copy(self) -> None:
        cred = JaxProbabilityIntervalsCredalSet(
            lower_bounds=jnp.array([[0.1, 0.2]]),
            upper_bounds=jnp.array([[0.5, 0.6]]),
        )
        arr = cred.numpy(force=True)
        assert arr.shape == (1, 2, 2)

    def test_num_classes(self) -> None:
        cred = JaxProbabilityIntervalsCredalSet(
            lower_bounds=jnp.array([[0.1, 0.2, 0.3]]),
            upper_bounds=jnp.array([[0.4, 0.5, 0.6]]),
        )
        assert cred.num_classes == 3


class TestEnsureJaxCategoricalDistribution:
    """The internal _ensure_jax_categorical_distribution coerces inputs."""

    def test_passthrough(self) -> None:
        from probly.representation.credal_set.jax import _ensure_jax_categorical_distribution  # noqa: PLC0415

        d = JaxProbabilityCategoricalDistribution(jnp.array([[0.5, 0.5]]))
        assert _ensure_jax_categorical_distribution(d) is d

    def test_wraps_array(self) -> None:
        from probly.representation.credal_set.jax import _ensure_jax_categorical_distribution  # noqa: PLC0415

        d = _ensure_jax_categorical_distribution(jnp.array([[0.5, 0.5]]))
        assert isinstance(d, JaxCategoricalDistribution)


class TestSampleProbabilities:
    """The internal _sample_probabilities helper rejects non-categorical samples."""

    def test_non_categorical_raises(self) -> None:
        from probly.representation.credal_set.jax import _sample_probabilities  # noqa: PLC0415

        sample = JaxArraySample(array=jnp.array([[0.5, 0.5], [0.3, 0.7]]), sample_axis=0)
        with pytest.raises(TypeError, match="JaxCategoricalDistribution"):
            _sample_probabilities(sample)


class TestCreateFromBounds:
    """`create_probability_intervals_from_lower_upper_array` and `_from_bounds`."""

    def test_create_intervals_from_packed_bounds(self) -> None:
        from probly.representation.credal_set._common import (  # noqa: PLC0415
            create_probability_intervals_from_lower_upper_array,
        )
        import probly.representation.credal_set.jax as _jax  # noqa: F401, PLC0415
        from probly.representation.credal_set.jax import JaxProbabilityIntervalsCredalSet  # noqa: PLC0415

        packed = jnp.array([[0.1, 0.2, 0.5, 0.6]])
        cred = create_probability_intervals_from_lower_upper_array(packed)
        assert isinstance(cred, JaxProbabilityIntervalsCredalSet)
        assert jnp.allclose(cred.lower_bounds, jnp.array([[0.1, 0.2]]))
        assert jnp.allclose(cred.upper_bounds, jnp.array([[0.5, 0.6]]))

    def test_create_intervals_from_bounds_separate_args(self) -> None:
        from probly.representation.credal_set._common import create_probability_intervals_from_bounds  # noqa: PLC0415
        import probly.representation.credal_set.jax as _jax  # noqa: F401, PLC0415
        from probly.representation.credal_set.jax import JaxProbabilityIntervalsCredalSet  # noqa: PLC0415

        probs = jnp.array([[0.4, 0.5]])
        lower = jnp.array([[0.1, 0.1]])
        upper = jnp.array([[0.1, 0.1]])
        cred = create_probability_intervals_from_bounds(probs, lower, upper)
        assert isinstance(cred, JaxProbabilityIntervalsCredalSet)


def test_jax_no_gradient_or_jit_import_side_effects() -> None:
    """Sanity check that jax_entropy / intersection_probability import cleanly (jit/grad-safe helpers)."""
    from probly.utils.jax import intersection_probability, jax_entropy  # noqa: PLC0415

    p = jnp.array([0.5, 0.5])
    assert jnp.isclose(jax.jit(jax_entropy)(p), jax_entropy(p))

    lower = jnp.array([0.1, 0.2])
    upper = jnp.array([0.5, 0.6])
    assert jnp.allclose(jax.jit(intersection_probability)(lower, upper), intersection_probability(lower, upper))
