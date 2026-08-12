"""Tests for the jax-backed conformal set classes."""

from __future__ import annotations

from jax import numpy as jnp
import pytest


class TestJaxArrayOneHotConformalSet:
    """Jax-backed one-hot conformal sets."""

    def test_from_bool_array(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayOneHotConformalSet  # noqa: PLC0415

        arr = jnp.array([[True, False, True], [False, True, False]])
        s = JaxArrayOneHotConformalSet(array=arr)
        assert jnp.array_equal(s.set_size, jnp.array([2, 1]))

    def test_from_int_array_with_only_zeros_and_ones(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayOneHotConformalSet  # noqa: PLC0415

        arr = jnp.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        s = JaxArrayOneHotConformalSet(array=arr)
        # Coerced to bool internally.
        assert s.array.dtype == bool
        assert jnp.array_equal(s.set_size, jnp.array([2, 1]))

    def test_invalid_array_raises(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayOneHotConformalSet  # noqa: PLC0415

        # Non-boolean / non-binary integer array -> rejected.
        with pytest.raises(ValueError, match="one-hot encoded"):
            JaxArrayOneHotConformalSet(array=jnp.array([[2, 1]], dtype=int))

    def test_from_array_sample_factory(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayOneHotConformalSet  # noqa: PLC0415

        arr = jnp.array([[True, False]])
        s = JaxArrayOneHotConformalSet.from_array_sample(arr)
        assert isinstance(s, JaxArrayOneHotConformalSet)

    def test_from_array_sample_with_non_array_raises(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayOneHotConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"jnp\.ndarray"):
            JaxArrayOneHotConformalSet.from_array_sample([[True, False]])  # type: ignore[arg-type]

    def test_from_sample_factory(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayOneHotConformalSet  # noqa: PLC0415
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        sample = JaxArraySample(array=jnp.array([[True, False]]), sample_axis=0)
        s = JaxArrayOneHotConformalSet.from_sample(sample)
        assert isinstance(s, JaxArrayOneHotConformalSet)


class TestJaxArrayIntervalConformalSet:
    """Jax-backed interval conformal sets."""

    def test_from_array_samples(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayIntervalConformalSet  # noqa: PLC0415

        lower = jnp.array([1.0, 2.0])
        upper = jnp.array([2.0, 3.0])
        s = JaxArrayIntervalConformalSet.from_array_samples(lower, upper)
        assert jnp.array_equal(s.set_size, jnp.array([1.0, 1.0]))

    def test_from_array_samples_non_array_raises(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"jnp\.ndarray"):
            JaxArrayIntervalConformalSet.from_array_samples([1, 2], jnp.array([2, 3]))  # type: ignore[arg-type]

    def test_from_samples_factory(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayIntervalConformalSet  # noqa: PLC0415
        from probly.representation.sample.jax import JaxArraySample  # noqa: PLC0415

        lower = JaxArraySample(array=jnp.array([1.0, 2.0]), sample_axis=0)
        upper = JaxArraySample(array=jnp.array([2.0, 3.0]), sample_axis=0)
        s = JaxArrayIntervalConformalSet.from_samples(lower, upper)
        assert jnp.array_equal(s.set_size, jnp.array([1.0, 1.0]))

    def test_from_samples_non_sample_raises(self) -> None:
        from probly.representation.conformal_set.jax import JaxArrayIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match="JaxArraySample"):
            JaxArrayIntervalConformalSet.from_samples(jnp.array([1.0]), jnp.array([2.0]))  # type: ignore[arg-type]
