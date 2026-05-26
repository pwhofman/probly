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


def _jax_modules():
    """Return (jax, jnp) or skip."""
    pytest.importorskip("jax")
    import jax as _jax  # noqa: PLC0415
    import jax.numpy as _jnp  # noqa: PLC0415

    return _jax, _jnp


class TestJaxArraySampleEdgeCases:
    """JaxArraySample validation and operations."""

    def test_invalid_sample_axis(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = jnp.zeros((2, 3))
        with pytest.raises(ValueError, match="out of bounds"):
            JaxArraySample(a, sample_axis=2)

    def test_negative_sample_axis_normalised(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = jnp.zeros((2, 3))
        s = JaxArraySample(a, sample_axis=-1)
        assert s.sample_axis == 1

    def test_negative_sample_axis_too_negative(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = jnp.zeros((2, 3))
        with pytest.raises(ValueError, match="out of bounds"):
            JaxArraySample(a, sample_axis=-3)

    def test_array_must_be_jax_array(self) -> None:
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        with pytest.raises(TypeError, match="JAX array"):
            JaxArraySample(np.zeros((2, 3)), sample_axis=0)  # type: ignore[arg-type]

    def test_weights_shape_mismatch(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        with pytest.raises(ValueError, match="weights must have shape"):
            JaxArraySample(jnp.zeros((2, 3)), sample_axis=0, weights=jnp.zeros(5))

    def test_T_property(self) -> None:  # noqa: N802
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        assert s.T.shape == (3, 2)

    def test_mT_property(self) -> None:  # noqa: N802
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        assert s.mT.shape == (3, 2)

    def test_size_property(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        assert s.size == 6

    def test_dtype_property(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(6, dtype=jnp.float32).reshape(2, 3), sample_axis=0)
        assert s.dtype == jnp.float32

    def test_device_property(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        assert s.device is not None

    def test_array_namespace(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        assert s.__array_namespace__() is not None

    def test_ndim_property(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.zeros((2, 3, 4)), sample_axis=0)
        assert s.ndim == 3

    def test_samples_property_moves_axis(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(24).reshape(2, 3, 4), sample_axis=2)
        assert s.samples.shape == (4, 2, 3)

    def test_samples_property_axis_zero(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(24).reshape(2, 3, 4), sample_axis=0)
        assert s.samples.shape == (2, 3, 4)

    def test_sample_size(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.zeros((5, 3)), sample_axis=0)
        assert s.sample_size == 5

    def test_sample_mean_unweighted(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(6, dtype=jnp.float32).reshape(2, 3), sample_axis=0)
        np.testing.assert_allclose(np.asarray(s.sample_mean()), [1.5, 2.5, 3.5])

    def test_sample_mean_weighted(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            sample_axis=0,
            weights=jnp.array([1.0, 0.0]),
        )
        np.testing.assert_allclose(np.asarray(s.sample_mean()), [1.0, 2.0])

    def test_sample_std_unweighted(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.array([1.0, 2.0, 3.0]), sample_axis=0)
        np.testing.assert_allclose(np.asarray(s.sample_std()), float(np.std([1.0, 2.0, 3.0])))

    def test_sample_std_weighted(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(
            jnp.array([1.0, 2.0, 3.0]),
            sample_axis=0,
            weights=jnp.array([1.0, 1.0, 1.0]),
        )
        np.testing.assert_allclose(np.asarray(s.sample_std()), float(np.std([1.0, 2.0, 3.0])))

    def test_sample_var_weighted_ddof_raises(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(
            jnp.array([1.0, 2.0, 3.0]),
            sample_axis=0,
            weights=jnp.array([1.0, 1.0, 1.0]),
        )
        with pytest.raises(ValueError, match="ddof > 0"):
            s.sample_var(ddof=1)

    def test_concat_two_jax_samples(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        b = JaxArraySample(jnp.arange(6, 12).reshape(2, 3), sample_axis=0)
        c = a.concat(b)
        assert c.array.shape == (4, 3)

    def test_concat_with_weights(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = JaxArraySample(
            jnp.arange(6).reshape(2, 3),
            sample_axis=0,
            weights=jnp.array([0.5, 0.5]),
        )
        b = JaxArraySample(
            jnp.arange(6, 12).reshape(2, 3),
            sample_axis=0,
        )
        c = a.concat(b)
        # b had no weights, so they get filled with ones.
        np.testing.assert_allclose(np.asarray(c.weights), [0.5, 0.5, 1.0, 1.0])

    def test_move_sample_axis(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = JaxArraySample(jnp.arange(24).reshape(2, 3, 4), sample_axis=0)
        moved = a.move_sample_axis(2)
        assert moved.sample_axis == 2
        assert moved.array.shape == (3, 4, 2)

    def test_array_dunder(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        np_a = np.asarray(a)
        assert isinstance(np_a, np.ndarray)

    def test_copy(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0, weights=jnp.array([0.5, 0.5]))
        c = a.copy()
        assert c.weights is not None
        np.testing.assert_array_equal(np.asarray(c.array), np.asarray(a.array))

    def test_to_device_same_device_returns_self(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        out = a.to_device(a.device)
        assert out is a

    def test_to_device_with_stream_raises(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        a = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        with pytest.raises(NotImplementedError, match="stream"):
            a.to_device(a.device, stream=1)

    def test_len(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample(jnp.arange(6).reshape(2, 3), sample_axis=0)
        assert len(s) == 2

    def test_from_iterable_auto_axis(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample.from_iterable(jnp.arange(12).reshape(3, 4))
        # auto -> -1
        assert s.sample_axis == 1

    def test_from_iterable_zero_dim_raises(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        with pytest.raises(ValueError, match="Cannot infer"):
            JaxArraySample.from_iterable(jnp.array(5))

    def test_from_iterable_empty_raises(self) -> None:
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        with pytest.raises(ValueError, match="Cannot infer"):
            JaxArraySample.from_iterable([])

    def test_from_iterable_with_dtype(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample.from_iterable(jnp.arange(12).reshape(3, 4), dtype=jnp.float32)
        assert s.array.dtype == jnp.float32

    def test_from_iterable_explicit_axis(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        s = JaxArraySample.from_iterable(jnp.arange(12).reshape(3, 4), sample_axis=0)
        assert s.sample_axis == 0
