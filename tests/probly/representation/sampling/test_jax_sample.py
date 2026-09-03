"""Tests for the JaxArraySample Representation."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp
import numpy as np

from probly.representation.jax_functions import (
    jax_average,
    jax_concatenate,
    jax_conj,
    jax_copy,
    jax_expand_dims,
    jax_matrix_transpose,
    jax_mean,
    jax_moveaxis,
    jax_reshape,
    jax_squeeze,
    jax_stack,
    jax_std,
    jax_sum,
    jax_swapaxes,
    jax_take_along_axis,
    jax_transpose,
    jax_var,
)
from probly.representation.jax_like import JaxLikeImplementation
from probly.representation.sample.array import ArraySample
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

    def test_negative_sample_axis_normalised_in_a_subclass(self) -> None:
        _, jnp = _jax_modules()
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        class _SubSample(JaxArraySample):
            pass

        assert _SubSample(jnp.zeros((2, 3)), sample_axis=-1).sample_axis == 1

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


@pytest.fixture
def sample_3d() -> JaxArraySample:
    """A 3-D sample whose sample axis sits in the middle."""
    return JaxArraySample(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4), sample_axis=1)


@pytest.fixture
def weighted_sample_2d() -> JaxArraySample:
    """A 2-D sample with weights along its sample axis."""
    return JaxArraySample(
        jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
        sample_axis=1,
        weights=jnp.array([0.1, 0.2, 0.3, 0.4]),
    )


class TestJaxArraySampleIsJaxLike:
    """The migration to ``JaxLikeImplementation`` and the members it now provides."""

    def test_is_jax_like_implementation(self, sample_3d: JaxArraySample) -> None:
        assert isinstance(sample_3d, JaxLikeImplementation)

    def test_is_usable_as_a_plain_jax_array(self, sample_3d: JaxArraySample) -> None:
        # The numpy backend's ArraySample works with plain numpy through ``__array__``; the jax
        # backend needs ``__jax_array__`` for the same interop, also for wrapped results like .T.
        np.testing.assert_allclose(np.asarray(jnp.sum(sample_3d)), np.asarray(jnp.sum(sample_3d.array)))
        np.testing.assert_allclose(
            np.asarray(jnp.sum(sample_3d.T)),
            np.asarray(jnp.sum(sample_3d.array)),
        )

        weights = jnp.ones(2)
        np.testing.assert_allclose(
            np.asarray(sample_3d.T @ weights),
            np.asarray(jnp.transpose(sample_3d.array) @ weights),
        )

    def test_from_iterable_of_a_traced_array_does_not_unroll(self) -> None:
        # A tracer does not satisfy the ``JaxLike`` protocol, so it used to be treated as an
        # iterable of rows.
        recorded: list[bool] = []

        def build(array: jax.Array) -> jax.Array:
            sample = JaxArraySample.from_iterable(array, sample_axis=0)
            recorded.append(sample.array is array)
            return sample.array

        traced = jax.jit(build)(jnp.zeros((4, 3)))

        assert recorded == [True]
        assert traced.shape == (4, 3)

    def test_transpose_property_tracks_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = sample_3d.T

        assert isinstance(result, JaxArraySample)
        assert result.shape == (4, 3, 2)
        assert result.sample_axis == 1

    def test_matrix_transpose_property_tracks_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = sample_3d.mT

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 4, 3)
        assert result.sample_axis == 2

    def test_adjoint_property_tracks_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = sample_3d.mH

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 2
        np.testing.assert_allclose(np.asarray(result), np.asarray(sample_3d.array).swapaxes(1, 2))

    def test_transpose_method_with_axes(self, sample_3d: JaxArraySample) -> None:
        result = sample_3d.transpose(2, 1, 0)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 1

    def test_at_exposes_indexed_update_helper(self, sample_3d: JaxArraySample) -> None:
        updated = sample_3d.at[0, 0, 0].set(99.0)

        assert isinstance(updated, jax.Array)
        assert float(updated[0, 0, 0]) == pytest.approx(99.0)

    def test_block_until_ready_returns_self(self, weighted_sample_2d: JaxArraySample) -> None:
        assert weighted_sample_2d.block_until_ready() is weighted_sample_2d

    def test_astype_casts_the_array(self, sample_3d: JaxArraySample) -> None:
        result = sample_3d.astype(jnp.float16)

        assert isinstance(result, JaxArraySample)
        assert result.array.dtype == jnp.float16
        assert result.sample_axis == sample_3d.sample_axis

    def test_getitem_tracks_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = sample_3d[0]

        assert isinstance(result, JaxArraySample)
        assert result.shape == (3, 4)
        assert result.sample_axis == 0

    def test_getitem_indexes_weights_on_the_sample_axis(self, weighted_sample_2d: JaxArraySample) -> None:
        result = weighted_sample_2d[:, jnp.array([3, 1])]

        assert isinstance(result, JaxArraySample)
        assert_weights_equal(result, jnp.array([0.4, 0.2]))

    def test_iteration_yields_samples(self, sample_3d: JaxArraySample) -> None:
        items = list(sample_3d)

        assert len(items) == 2
        assert all(isinstance(item, JaxArraySample) for item in items)

    def test_setitem_is_rejected(self, sample_3d: JaxArraySample) -> None:
        with pytest.raises(TypeError, match="immutable"):
            sample_3d[0] = 1.0


class TestJaxFunctionReductions:
    """``__jax_function__`` keeps the sample axis in sync across reductions."""

    @pytest.mark.parametrize("func", [jax_mean, jax_sum, jax_std, jax_var, jax_average])
    def test_reduction_over_other_axis_shifts_sample_axis(self, sample_3d: JaxArraySample, func: object) -> None:
        result = func(sample_3d, 0)  # ty: ignore[call-non-callable]

        assert isinstance(result, JaxArraySample)
        assert result.shape == (3, 4)
        assert result.sample_axis == 0

    @pytest.mark.parametrize("func", [jax_mean, jax_sum, jax_std, jax_var, jax_average])
    def test_reduction_over_sample_axis_drops_the_wrapper(self, sample_3d: JaxArraySample, func: object) -> None:
        result = func(sample_3d, 1)  # ty: ignore[call-non-callable]

        assert not isinstance(result, JaxArraySample)
        assert result.shape == (2, 4)

    def test_reduction_over_multiple_axes(self, sample_3d: JaxArraySample) -> None:
        result = jax_sum(sample_3d, (0, 2))

        assert isinstance(result, JaxArraySample)
        assert result.shape == (3,)
        assert result.sample_axis == 0

    def test_reduction_over_negative_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_mean(sample_3d, -1)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 1

    def test_keepdims_keeps_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_mean(sample_3d, 0, keepdims=True)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (1, 3, 4)
        assert result.sample_axis == 1

    def test_reduction_over_all_axes_drops_the_wrapper(self, sample_3d: JaxArraySample) -> None:
        result = jax_sum(sample_3d)

        assert not isinstance(result, JaxArraySample)

    def test_average_ignores_the_sample_weights(self, weighted_sample_2d: JaxArraySample) -> None:
        # Mirrors the numpy backend, which never injects the stored weights into np.average.
        result = jax_average(weighted_sample_2d, 1)

        expected = jnp.average(weighted_sample_2d.array, axis=1)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected))

    def test_average_over_all_axes_ignores_the_sample_weights(self, weighted_sample_2d: JaxArraySample) -> None:
        result = jax_average(weighted_sample_2d, None)

        np.testing.assert_allclose(np.asarray(result), np.asarray(jnp.average(weighted_sample_2d.array)))

    def test_sample_mean_uses_the_sample_weights(self, weighted_sample_2d: JaxArraySample) -> None:
        result = weighted_sample_2d.sample_mean()

        expected = jnp.average(weighted_sample_2d.array, axis=1, weights=weighted_sample_2d.weights)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected))

    def test_average_keeps_explicit_weights(self, weighted_sample_2d: JaxArraySample) -> None:
        weights = jnp.array([1.0, 0.0, 0.0, 0.0])

        result = jax_average(weighted_sample_2d, 1, weights)

        np.testing.assert_allclose(np.asarray(result), np.asarray(weighted_sample_2d.array[:, 0]))

    def test_reduction_preserves_weights_off_the_sample_axis(self, weighted_sample_2d: JaxArraySample) -> None:
        result = jax_mean(weighted_sample_2d, 0)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 0
        assert_weights_equal(result, weighted_sample_2d.weights)

    def test_std_and_var_honour_ddof(self, sample_3d: JaxArraySample) -> None:
        std = jax_std(sample_3d, 0, ddof=1)
        var = jax_var(sample_3d, 0, ddof=1)

        np.testing.assert_allclose(np.asarray(std), np.asarray(jnp.std(sample_3d.array, axis=0, ddof=1)), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(var), np.asarray(jnp.var(sample_3d.array, axis=0, ddof=1)), rtol=1e-6)


class TestJaxFunctionAxisMoves:
    """``__jax_function__`` for the axis-permuting wrappers."""

    def test_transpose_without_axes(self, sample_3d: JaxArraySample) -> None:
        result = jax_transpose(sample_3d)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (4, 3, 2)
        assert result.sample_axis == 1

    def test_transpose_with_axes(self, sample_3d: JaxArraySample) -> None:
        result = jax_transpose(sample_3d, (1, 2, 0))

        assert isinstance(result, JaxArraySample)
        assert result.shape == (3, 4, 2)
        assert result.sample_axis == 0

    def test_transpose_with_negative_axes(self, sample_3d: JaxArraySample) -> None:
        result = jax_transpose(sample_3d, (-2, -1, -3))

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 0

    def test_matrix_transpose(self, sample_3d: JaxArraySample) -> None:
        result = jax_matrix_transpose(sample_3d)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 2

    def test_matrix_transpose_leaves_other_axes_alone(self) -> None:
        sample = JaxArraySample(jnp.zeros((2, 3, 4)), sample_axis=0)

        result = jax_matrix_transpose(sample)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 0

    def test_moveaxis_moves_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_moveaxis(sample_3d, 1, 2)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 4, 3)
        assert result.sample_axis == 2

    def test_moveaxis_of_another_axis_shifts_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_moveaxis(sample_3d, 0, 2)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (3, 4, 2)
        assert result.sample_axis == 0

    def test_moveaxis_with_negative_axes(self, sample_3d: JaxArraySample) -> None:
        result = jax_moveaxis(sample_3d, -2, 0)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 0

    def test_moveaxis_with_sequences(self, sample_3d: JaxArraySample) -> None:
        result = jax_moveaxis(sample_3d, (0, 1), (2, 0))

        assert isinstance(result, JaxArraySample)
        assert result.shape == (3, 4, 2)
        assert result.sample_axis == 0

    def test_conj_preserves_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_conj(sample_3d)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 1


class TestJaxFunctionShapeChanges:
    """``__jax_function__`` for the shape-changing wrappers, mirroring the numpy backend."""

    def test_copy_preserves_the_sample_axis(self, weighted_sample_2d: JaxArraySample) -> None:
        result = jax_copy(weighted_sample_2d)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == weighted_sample_2d.sample_axis
        assert_weights_equal(result, weighted_sample_2d.weights)

    def test_reshape_keeps_a_surviving_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_reshape(sample_3d, (2, 3, 2, 2))

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 3, 2, 2)
        assert result.sample_axis == 1

    def test_reshape_drops_a_merged_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_reshape(sample_3d, (2, 12))

        assert not isinstance(result, JaxArraySample)
        assert result.shape == (2, 12)

    def test_reshape_matches_the_numpy_backend(self, sample_3d: JaxArraySample) -> None:
        array_sample = ArraySample(np.asarray(sample_3d.array), sample_axis=sample_3d.sample_axis)

        result = jax_reshape(sample_3d, (2, 3, 4, 1))
        expected = np.reshape(array_sample, (2, 3, 4, 1))

        assert isinstance(result, JaxArraySample)
        assert isinstance(expected, ArraySample)
        assert result.sample_axis == expected.sample_axis

    def test_flatten_returns_a_bare_array(self, sample_3d: JaxArraySample) -> None:
        assert not isinstance(sample_3d.flatten(), JaxArraySample)
        assert not isinstance(sample_3d.ravel(), JaxArraySample)

    def test_squeeze_shifts_the_sample_axis(self) -> None:
        sample = JaxArraySample(jnp.zeros((1, 2, 3)), sample_axis=1)

        result = jax_squeeze(sample, 0)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 3)
        assert result.sample_axis == 0

    def test_squeeze_of_the_sample_axis_drops_the_wrapper(self) -> None:
        sample = JaxArraySample(jnp.zeros((2, 1, 3)), sample_axis=1)

        result = jax_squeeze(sample)

        assert not isinstance(result, JaxArraySample)
        assert result.shape == (2, 3)

    def test_expand_dims_shifts_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_expand_dims(sample_3d, 0)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (1, 2, 3, 4)
        assert result.sample_axis == 2

    def test_expand_dims_after_the_sample_axis_keeps_it(self, sample_3d: JaxArraySample) -> None:
        result = jax_expand_dims(sample_3d, -1)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 1

    def test_swapaxes_follows_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_swapaxes(sample_3d, 1, 2)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 4, 3)
        assert result.sample_axis == 2

    def test_swapaxes_of_other_axes_keeps_the_sample_axis(self) -> None:
        sample = JaxArraySample(jnp.zeros((2, 3, 4)), sample_axis=1)

        result = jax_swapaxes(sample, 0, -1)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 1

    def test_take_along_axis_off_the_sample_axis_keeps_it(self, sample_3d: JaxArraySample) -> None:
        indices = jnp.zeros((2, 3, 1), dtype=jnp.int32)

        result = jax_take_along_axis(sample_3d, indices, axis=-1)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 3, 1)
        assert result.sample_axis == 1

    def test_take_along_axis_on_the_sample_axis_drops_the_wrapper(self, sample_3d: JaxArraySample) -> None:
        indices = jnp.zeros((2, 1, 4), dtype=jnp.int32)

        result = jax_take_along_axis(sample_3d, indices, axis=1)

        assert not isinstance(result, JaxArraySample)
        assert result.shape == (2, 1, 4)


class TestJaxFunctionSequences:
    """``__jax_function__`` for concatenate and stack."""

    def test_concatenate_along_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_concatenate([sample_3d, sample_3d], 1)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 6, 4)
        assert result.sample_axis == 1

    def test_concatenate_along_another_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_concatenate([sample_3d, sample_3d], 0)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (4, 3, 4)
        assert result.sample_axis == 1

    def test_concatenate_with_a_plain_array(self, sample_3d: JaxArraySample) -> None:
        result = jax_concatenate([sample_3d, sample_3d.array], 1)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 6, 4)

    def test_concatenate_with_mixed_sample_axes_returns_a_plain_array(self, sample_3d: JaxArraySample) -> None:
        other = JaxArraySample(sample_3d.array, sample_axis=0)

        result = jax_concatenate([sample_3d, other], 0)

        assert not isinstance(result, JaxArraySample)

    def test_concatenate_combines_weights(self, weighted_sample_2d: JaxArraySample) -> None:
        result = jax_concatenate([weighted_sample_2d, weighted_sample_2d], 1)

        assert isinstance(result, JaxArraySample)
        assert_weights_equal(result, jnp.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]))

    def test_concatenate_fills_missing_weights_with_ones(self, weighted_sample_2d: JaxArraySample) -> None:
        unweighted = JaxArraySample(weighted_sample_2d.array, sample_axis=1)

        result = jax_concatenate([weighted_sample_2d, unweighted], 1)

        assert_weights_equal(result, jnp.array([0.1, 0.2, 0.3, 0.4, 1.0, 1.0, 1.0, 1.0]))

    def test_concatenate_of_weighted_samples_off_the_sample_axis_raises(
        self, weighted_sample_2d: JaxArraySample
    ) -> None:
        with pytest.raises(ValueError, match="only support concatenate along the sample axis"):
            jax_concatenate([weighted_sample_2d, weighted_sample_2d], 0)

    def test_stack_before_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_stack([sample_3d, sample_3d], 0)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 2, 3, 4)
        assert result.sample_axis == 2

    def test_stack_after_the_sample_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_stack([sample_3d, sample_3d], 2)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 3, 2, 4)
        assert result.sample_axis == 1

    def test_stack_with_negative_axis(self, sample_3d: JaxArraySample) -> None:
        result = jax_stack([sample_3d, sample_3d], -1)

        assert isinstance(result, JaxArraySample)
        assert result.shape == (2, 3, 4, 2)
        assert result.sample_axis == 1

    def test_stack_of_weighted_samples_raises(self, weighted_sample_2d: JaxArraySample) -> None:
        with pytest.raises(ValueError, match="do not support stack"):
            jax_stack([weighted_sample_2d, weighted_sample_2d], 0)


class TestJaxArraySampleConversions:
    """``__jax_like__`` and ``__array_like__`` round-trips."""

    def test_array_sample_to_jax_sample(self) -> None:
        array_sample = ArraySample(
            np.arange(12, dtype=np.float32).reshape(3, 4),
            sample_axis=1,
            weights=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        )

        converted = array_sample.__jax_like__()

        assert isinstance(converted, JaxArraySample)
        assert converted.sample_axis == 1
        assert_weights_equal(converted, array_sample.weights)
        np.testing.assert_allclose(np.asarray(converted), np.asarray(array_sample))

    def test_array_sample_to_jax_sample_honours_dtype(self) -> None:
        array_sample = ArraySample(np.arange(12).reshape(3, 4), sample_axis=1)

        converted = array_sample.__jax_like__(jnp.float32)

        assert converted.dtype == jnp.float32

    def test_round_trip_back_to_array_sample(self) -> None:
        array_sample = ArraySample(
            np.arange(12, dtype=np.float32).reshape(3, 4),
            sample_axis=1,
            weights=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        )

        round_tripped = array_sample.__jax_like__().__array_like__()

        assert isinstance(round_tripped, ArraySample)
        assert round_tripped.sample_axis == 1
        np.testing.assert_allclose(round_tripped.array, array_sample.array)
        np.testing.assert_allclose(round_tripped.weights, array_sample.weights)

    def test_jax_like_without_arguments_returns_self(self, sample_3d: JaxArraySample) -> None:
        assert sample_3d.__jax_like__() is sample_3d

    def test_jax_like_with_dtype_casts(self, sample_3d: JaxArraySample) -> None:
        converted = sample_3d.__jax_like__(jnp.float16)

        assert isinstance(converted, JaxArraySample)
        assert converted.dtype == jnp.float16
        assert converted.sample_axis == sample_3d.sample_axis


class TestJaxArraySamplePytree:
    """``JaxArraySample`` participates in JAX transformations as a pytree."""

    def test_tree_map_round_trip(self, sample_3d: JaxArraySample) -> None:
        mapped = jax.tree.map(lambda leaf: leaf * 2, sample_3d)

        assert isinstance(mapped, JaxArraySample)
        assert mapped.sample_axis == sample_3d.sample_axis
        np.testing.assert_allclose(np.asarray(mapped), np.asarray(sample_3d) * 2)

    def test_tree_map_keeps_weights_as_children(self, weighted_sample_2d: JaxArraySample) -> None:
        leaves = jax.tree.leaves(weighted_sample_2d)

        assert len(leaves) == 2

    def test_tree_flatten_puts_the_sample_axis_in_aux_data(self, sample_3d: JaxArraySample) -> None:
        children, aux = sample_3d.tree_flatten()

        assert children[0] is sample_3d.array
        assert ("sample_axis", 1) in aux[1]

    def test_jit_round_trip(self, sample_3d: JaxArraySample) -> None:
        @jax.jit
        def double(sample: JaxArraySample) -> JaxArraySample:
            return JaxArraySample(sample.array * 2, sample.sample_axis)

        result = double(sample_3d)

        assert isinstance(result, JaxArraySample)
        assert result.sample_axis == 1
        np.testing.assert_allclose(np.asarray(result), np.asarray(sample_3d) * 2)

    def test_vmap_over_a_sample(self, sample_3d: JaxArraySample) -> None:
        result = jax.vmap(lambda sample: sample.array.sum())(sample_3d)

        assert result.shape == (2,)
