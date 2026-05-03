"""Tests for the JaxArraySample Representation."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
from jax import numpy as jnp
import numpy as np

from probly.representation.sample.jax import JaxArraySample


def assert_weights_equal(sample: JaxArraySample, expected: object) -> None:
    assert sample.weights is not None
    assert np.array_equal(np.asarray(sample.weights), np.asarray(expected))


class TestJaxArraySample:
    def test_sample_internal_array(self, jax_array_sample_2d: JaxArraySample) -> None:
        assert isinstance(jax_array_sample_2d.array, jnp.ndarray)

    def test_sample_length(self, jax_array_sample_2d: JaxArraySample) -> None:
        assert len(jax_array_sample_2d) == len(jax_array_sample_2d.array)

    def test_sample_ndim(self, jax_array_sample_2d: JaxArraySample) -> None:
        assert jax_array_sample_2d.ndim == 2

    def test_sample_shape(self, jax_array_sample_2d: JaxArraySample) -> None:
        assert jax_array_sample_2d.shape == jax_array_sample_2d.array.shape

    def test_sample_move_axis(self, jax_array_sample_2d: JaxArraySample) -> None:
        moved_sample = jax_array_sample_2d.move_sample_axis(0)
        assert isinstance(moved_sample, JaxArraySample)
        assert moved_sample.sample_axis == 0
        assert (
            jax_array_sample_2d.shape[jax_array_sample_2d.sample_axis] == moved_sample.shape[moved_sample.sample_axis]
        )

    def test_sample_concat(self, jax_array_sample_2d: JaxArraySample) -> None:
        res = jax_array_sample_2d.concat(jax_array_sample_2d.move_sample_axis(0))
        assert isinstance(res, JaxArraySample)
        assert res.sample_axis == jax_array_sample_2d.sample_axis
        assert res.sample_size == 2 * jax_array_sample_2d.sample_size

    def test_from_iterable_preserves_weights(self) -> None:
        weights = jnp.array([0.1, 0.2, 0.3])

        sample = JaxArraySample.from_iterable(jnp.arange(6).reshape((3, 2)), sample_axis=0, weights=weights)

        assert_weights_equal(sample, weights)

    def test_constructor_rejects_wrong_weight_shape(self) -> None:
        with pytest.raises(ValueError, match="weights must have shape"):
            JaxArraySample(jnp.arange(12).reshape((3, 4)), sample_axis=1, weights=jnp.array([0.1, 0.2, 0.3]))

    def test_from_sample_preserves_weights(self) -> None:
        weights = jnp.array([0.1, 0.2, 0.3, 0.4])
        sample = JaxArraySample(jnp.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        converted = JaxArraySample.from_sample(sample, sample_axis=0)

        assert converted.sample_axis == 0
        assert_weights_equal(converted, weights)

    def test_copy_preserves_weights(self) -> None:
        weights = jnp.array([0.1, 0.2, 0.3, 0.4])
        sample = JaxArraySample(jnp.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        copied = sample.copy()

        assert copied is not sample
        assert_weights_equal(copied, weights)

    def test_sample_move_axis_preserves_weights(self) -> None:
        weights = jnp.array([0.1, 0.2, 0.3, 0.4])
        sample = JaxArraySample(jnp.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        moved_sample = sample.move_sample_axis(0)

        assert moved_sample.sample_axis == 0
        assert_weights_equal(moved_sample, weights)

    def test_sample_concat_combines_weights(self) -> None:
        left = JaxArraySample(jnp.arange(12).reshape((3, 4)), sample_axis=1, weights=jnp.array([0.1, 0.2, 0.3, 0.4]))
        right = JaxArraySample(
            jnp.arange(12, 24).reshape((4, 3)), sample_axis=0, weights=jnp.array([0.5, 0.6, 0.7, 0.8])
        )

        result = left.concat(right)

        assert result.sample_axis == 1
        assert_weights_equal(result, jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))

    def test_sample_concat_fills_missing_weights_with_ones(self) -> None:
        left = JaxArraySample(jnp.arange(12).reshape((3, 4)), sample_axis=1)
        right = JaxArraySample(
            jnp.arange(12, 24).reshape((3, 4)), sample_axis=1, weights=jnp.array([0.5, 0.6, 0.7, 0.8])
        )

        result = left.concat(right)

        assert_weights_equal(result, jnp.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.6, 0.7, 0.8]))

    def test_sample_mean_uses_weights(self) -> None:
        weights = jnp.array([0.1, 0.2, 0.3, 0.4])
        sample = JaxArraySample(jnp.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        result = sample.sample_mean()

        assert np.allclose(np.asarray(result), np.asarray(jnp.average(sample.array, axis=1, weights=weights)))

    def test_sample_var_and_std_use_weights(self) -> None:
        weights = jnp.array([0.1, 0.2, 0.3, 0.4])
        sample = JaxArraySample(jnp.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)
        average = jnp.average(sample.array, axis=1, weights=weights, keepdims=True)
        expected_var = jnp.average((sample.array - average) ** 2, axis=1, weights=weights)

        assert np.allclose(np.asarray(sample.sample_var()), np.asarray(expected_var))
        assert np.allclose(np.asarray(sample.sample_std()), np.asarray(jnp.sqrt(expected_var)))

    def test_weighted_sample_var_rejects_ddof(self) -> None:
        sample = JaxArraySample(jnp.arange(12).reshape((3, 4)), sample_axis=1, weights=jnp.ones(4))

        with pytest.raises(ValueError, match="ddof"):
            sample.sample_var(ddof=1)


class TestJaxArraySampleProtectedAxis:
    """Regression tests for JaxArraySample wrapping JaxAxisProtected types."""

    def test_protected_axis_distribution_at_nonzero_sample_axis(self) -> None:
        """Regression: JaxArraySample wrapping a JaxAxisProtected at sample_axis!=0 must dispatch correctly.

        Previously ``JaxArraySample.samples`` called ``jnp.moveaxis`` on
        ``self.array`` directly, which fails for protected-axis wrappers
        because JAX has no ``__array_function__`` analog. The protocol
        ``__instancehook__`` reading ``samples`` would crash and tear down
        the entire ``Flexdispatch`` lookup.
        """
        from probly.quantification.measure.distribution import (  # noqa: PLC0415
            entropy_of_expected_predictive_distribution,
        )
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxCategoricalDistribution,
            JaxCategoricalDistributionSample,
        )

        probs = jnp.arange(5 * 3 * 4, dtype=jnp.float32).reshape(5, 3, 4) + 0.01
        probs = probs / probs.sum(axis=-1, keepdims=True)
        dist = JaxCategoricalDistribution(probs)

        sample = JaxArraySample(array=dist, sample_axis=1)

        # Protocol check must not raise:
        assert isinstance(sample, JaxCategoricalDistributionSample)

        # Dispatch through the measure flexdispatch must produce a finite array.
        # Categorical batch shape is (5,) after rotating sample_axis=1 to 0 and
        # taking the mean over the sample axis; entropy further reduces the class axis.
        result = entropy_of_expected_predictive_distribution(sample)
        assert result.shape == (5,)
        assert bool(jnp.all(jnp.isfinite(result)))

    def test_gaussian_protected_axis_at_nonzero_sample_axis(self) -> None:
        """Regression: same fix path for ``JaxGaussianDistribution``.

        Constructs a ``JaxArraySample`` whose ``array`` is a Gaussian
        distribution wrapper and verifies the structural protocol check
        does not crash and the rotated samples preserve the wrapper type.
        """
        from probly.representation.distribution.jax_gaussian import (  # noqa: PLC0415
            JaxGaussianDistribution,
            JaxGaussianDistributionSample,
        )

        mean = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
        var = jnp.ones_like(mean)
        dist = JaxGaussianDistribution(mean=mean, var=var)

        sample = JaxArraySample(array=dist, sample_axis=1)

        # Protocol check must not raise.
        assert isinstance(sample, JaxGaussianDistributionSample)

        rotated = sample.samples
        assert isinstance(rotated, JaxGaussianDistribution)
        # After rotating sample_axis=1 -> 0, the underlying mean/var have shape (3, 2, 4).
        assert rotated.mean.shape == (3, 2, 4)
        assert rotated.var.shape == (3, 2, 4)
