"""Tests for the jax-backed conformal set classes."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
from jax import numpy as jnp


class TestJaxOneHotConformalSet:
    """Jax-backed one-hot conformal sets."""

    def test_from_bool_array(self) -> None:
        from probly.representation.conformal_set.jax import JaxOneHotConformalSet  # noqa: PLC0415

        arr = jnp.array([[True, False, True], [False, True, False]])
        s = JaxOneHotConformalSet(array=arr)
        assert jnp.array_equal(s.set_size, jnp.array([2, 1]))

    def test_from_int_array_with_only_zeros_and_ones(self) -> None:
        from probly.representation.conformal_set.jax import JaxOneHotConformalSet  # noqa: PLC0415

        arr = jnp.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        s = JaxOneHotConformalSet(array=arr)
        # Coerced to bool internally.
        assert s.array.dtype == bool
        assert jnp.array_equal(s.set_size, jnp.array([2, 1]))

    def test_invalid_array_raises(self) -> None:
        from probly.representation.conformal_set.jax import JaxOneHotConformalSet  # noqa: PLC0415

        # Non-boolean / non-binary integer array -> rejected.
        with pytest.raises(ValueError, match="one-hot encoded"):
            JaxOneHotConformalSet(array=jnp.array([[2, 1]], dtype=int))

    def test_from_array_sample_factory(self) -> None:
        from probly.representation.conformal_set.jax import JaxOneHotConformalSet  # noqa: PLC0415

        arr = jnp.array([[True, False]])
        s = JaxOneHotConformalSet.from_numpy_sample(arr)
        assert isinstance(s, JaxOneHotConformalSet)

    def test_from_array_sample_with_non_array_raises(self) -> None:
        from probly.representation.conformal_set.jax import JaxOneHotConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"jnp\.ndarray"):
            JaxOneHotConformalSet.from_numpy_sample([[True, False]])  # type: ignore[arg-type]

    def test_from_sample_factory(self) -> None:
        from probly.representation.conformal_set.jax import JaxOneHotConformalSet  # noqa: PLC0415
        from probly.representation.sample.jax import JaxSample  # noqa: PLC0415

        sample = JaxSample(array=jnp.array([[True, False]]), sample_axis=0)
        s = JaxOneHotConformalSet.from_sample(sample)
        assert isinstance(s, JaxOneHotConformalSet)


class TestJaxIntervalConformalSet:
    """Jax-backed interval conformal sets."""

    def test_from_array_samples(self) -> None:
        from probly.representation.conformal_set.jax import JaxIntervalConformalSet  # noqa: PLC0415

        lower = jnp.array([1.0, 2.0])
        upper = jnp.array([2.0, 3.0])
        s = JaxIntervalConformalSet.from_numpy_samples(lower, upper)
        assert jnp.array_equal(s.set_size, jnp.array([1.0, 1.0]))

    def test_from_array_samples_non_array_raises(self) -> None:
        from probly.representation.conformal_set.jax import JaxIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"jnp\.ndarray"):
            JaxIntervalConformalSet.from_numpy_samples([1, 2], jnp.array([2, 3]))  # type: ignore[arg-type]

    def test_from_samples_factory(self) -> None:
        from probly.representation.conformal_set.jax import JaxIntervalConformalSet  # noqa: PLC0415
        from probly.representation.sample.jax import JaxSample  # noqa: PLC0415

        lower = JaxSample(array=jnp.array([1.0, 2.0]), sample_axis=0)
        upper = JaxSample(array=jnp.array([2.0, 3.0]), sample_axis=0)
        s = JaxIntervalConformalSet.from_samples(lower, upper)
        assert jnp.array_equal(s.set_size, jnp.array([1.0, 1.0]))

    def test_from_samples_non_sample_raises(self) -> None:
        from probly.representation.conformal_set.jax import JaxIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match="JaxSample"):
            JaxIntervalConformalSet.from_samples(jnp.array([1.0]), jnp.array([2.0]))  # type: ignore[arg-type]
