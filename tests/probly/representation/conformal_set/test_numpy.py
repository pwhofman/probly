"""Tests for the numpy-backed conformal set classes."""

from __future__ import annotations

import numpy as np
import pytest


class TestNumpyOneHotConformalSet:
    """Numpy-backed one-hot conformal sets."""

    def test_from_bool_array(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyOneHotConformalSet  # noqa: PLC0415

        arr = np.array([[True, False, True], [False, True, False]])
        s = NumpyOneHotConformalSet(array=arr)
        np.testing.assert_array_equal(s.set_size, [2, 1])

    def test_from_int_array_with_only_zeros_and_ones(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyOneHotConformalSet  # noqa: PLC0415

        arr = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
        s = NumpyOneHotConformalSet(array=arr)
        # Coerced to bool internally.
        assert s.array.dtype == bool
        np.testing.assert_array_equal(s.set_size, [2, 1])

    def test_invalid_array_raises(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyOneHotConformalSet  # noqa: PLC0415

        # Non-boolean / non-binary integer array -> rejected.
        with pytest.raises(ValueError, match="one-hot encoded"):
            NumpyOneHotConformalSet(array=np.array([[2, 1]], dtype=int))

    def test_from_array_factory(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyOneHotConformalSet  # noqa: PLC0415

        arr = np.array([[True, False]])
        s = NumpyOneHotConformalSet.from_array(arr)
        assert isinstance(s, NumpyOneHotConformalSet)

    def test_from_array_with_non_array_raises(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyOneHotConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"np\.ndarray"):
            NumpyOneHotConformalSet.from_array([[True, False]])  # type: ignore[arg-type]

    def test_from_sample_factory(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyOneHotConformalSet  # noqa: PLC0415
        from probly.representation.sample.numpy import NumpySample  # noqa: PLC0415

        sample = NumpySample(array=np.array([[True, False]]), sample_axis=0)
        s = NumpyOneHotConformalSet.from_sample(sample)
        assert isinstance(s, NumpyOneHotConformalSet)


class TestNumpyIntervalConformalSet:
    """Numpy-backed interval conformal sets."""

    def test_from_arrays(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyIntervalConformalSet  # noqa: PLC0415

        lower = np.array([1.0, 2.0])
        upper = np.array([2.0, 3.0])
        s = NumpyIntervalConformalSet.from_arrays(lower, upper)
        np.testing.assert_array_equal(s.set_size, [1.0, 1.0])

    def test_from_arrays_non_array_raises(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"np\.ndarray"):
            NumpyIntervalConformalSet.from_arrays([1, 2], np.array([2, 3]))  # type: ignore[arg-type]

    def test_from_samples_factory(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyIntervalConformalSet  # noqa: PLC0415
        from probly.representation.sample.numpy import NumpySample  # noqa: PLC0415

        lower = NumpySample(array=np.array([1.0, 2.0]), sample_axis=0)
        upper = NumpySample(array=np.array([2.0, 3.0]), sample_axis=0)
        s = NumpyIntervalConformalSet.from_samples(lower, upper)
        np.testing.assert_array_equal(s.set_size, [1.0, 1.0])

    def test_from_samples_non_sample_raises(self) -> None:
        from probly.representation.conformal_set.numpy import NumpyIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match="NumpySample"):
            NumpyIntervalConformalSet.from_samples(np.array([1.0]), np.array([2.0]))  # type: ignore[arg-type]
