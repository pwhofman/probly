"""Tests for the numpy-backed conformal set classes."""

from __future__ import annotations

import numpy as np
import pytest


class TestArrayOneHotConformalSet:
    """Numpy-backed one-hot conformal sets."""

    def test_from_bool_array(self) -> None:
        from probly.representation.conformal_set.array import ArrayOneHotConformalSet  # noqa: PLC0415

        arr = np.array([[True, False, True], [False, True, False]])
        s = ArrayOneHotConformalSet(array=arr)
        np.testing.assert_array_equal(s.set_size, [2, 1])

    def test_from_int_array_with_only_zeros_and_ones(self) -> None:
        from probly.representation.conformal_set.array import ArrayOneHotConformalSet  # noqa: PLC0415

        arr = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        s = ArrayOneHotConformalSet(array=arr)
        # Coerced to bool internally.
        assert s.array.dtype == bool
        np.testing.assert_array_equal(s.set_size, [2, 1])

    def test_invalid_array_raises(self) -> None:
        from probly.representation.conformal_set.array import ArrayOneHotConformalSet  # noqa: PLC0415

        # Non-boolean / non-binary integer array -> rejected.
        with pytest.raises(ValueError, match="one-hot encoded"):
            ArrayOneHotConformalSet(array=np.array([[2, 1]], dtype=int))

    def test_from_array_sample_factory(self) -> None:
        from probly.representation.conformal_set.array import ArrayOneHotConformalSet  # noqa: PLC0415

        arr = np.array([[True, False]])
        s = ArrayOneHotConformalSet.from_array_sample(arr)
        assert isinstance(s, ArrayOneHotConformalSet)

    def test_from_array_sample_with_non_array_raises(self) -> None:
        from probly.representation.conformal_set.array import ArrayOneHotConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"np\.ndarray"):
            ArrayOneHotConformalSet.from_array_sample([[True, False]])  # type: ignore[arg-type]

    def test_from_sample_factory(self) -> None:
        from probly.representation.conformal_set.array import ArrayOneHotConformalSet  # noqa: PLC0415
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.array([[True, False]]), sample_axis=0)
        s = ArrayOneHotConformalSet.from_sample(sample)
        assert isinstance(s, ArrayOneHotConformalSet)


class TestArrayIntervalConformalSet:
    """Numpy-backed interval conformal sets."""

    def test_from_array_samples(self) -> None:
        from probly.representation.conformal_set.array import ArrayIntervalConformalSet  # noqa: PLC0415

        lower = np.array([1.0, 2.0])
        upper = np.array([2.0, 3.0])
        s = ArrayIntervalConformalSet.from_array_samples(lower, upper)
        np.testing.assert_array_equal(s.set_size, [1.0, 1.0])

    def test_from_array_samples_non_array_raises(self) -> None:
        from probly.representation.conformal_set.array import ArrayIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"np\.ndarray"):
            ArrayIntervalConformalSet.from_array_samples([1, 2], np.array([2, 3]))  # type: ignore[arg-type]

    def test_from_samples_factory(self) -> None:
        from probly.representation.conformal_set.array import ArrayIntervalConformalSet  # noqa: PLC0415
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        lower = ArraySample(array=np.array([1.0, 2.0]), sample_axis=0)
        upper = ArraySample(array=np.array([2.0, 3.0]), sample_axis=0)
        s = ArrayIntervalConformalSet.from_samples(lower, upper)
        np.testing.assert_array_equal(s.set_size, [1.0, 1.0])

    def test_from_samples_non_sample_raises(self) -> None:
        from probly.representation.conformal_set.array import ArrayIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match="ArraySample"):
            ArrayIntervalConformalSet.from_samples(np.array([1.0]), np.array([2.0]))  # type: ignore[arg-type]
