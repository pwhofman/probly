"""NumPy sample ufunc outputs retain metadata without replacing output buffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, cast

from ndlike.protected_axis.numpy import NumpyAxisProtected
import numpy as np
import pytest

from probly.representation.sample.numpy import NumpySample


@dataclass(frozen=True, eq=False)
class _NumpyVectors(NumpyAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        np.modf: ["__call__"],
        np.frexp: ["__call__"],
        np.divmod: ["__call__"],
    }


@pytest.fixture
def sample():
    return NumpySample(
        np.array([[1.25, 2.5, 3.75], [4.25, 5.5, 6.75]]),
        sample_axis=1,
        weights=np.array([1.0, 2.0, 3.0]),
    )


@pytest.mark.parametrize("ufunc", [np.modf, np.frexp, np.divmod])
@pytest.mark.parametrize("first_out", ["allocate", "array", "sample"])
@pytest.mark.parametrize("second_out", ["allocate", "array", "sample"])
def test_multiple_outputs_preserve_metadata_and_output_identity(sample, ufunc, first_out, second_out):
    args = (sample, 2) if ufunc.nin == 2 else (sample,)
    native_args = (sample.array, 2) if ufunc.nin == 2 else (sample.array,)
    expected = ufunc(*native_args)
    outs = []
    for kind, value in zip((first_out, second_out), expected, strict=True):
        buffer = np.empty_like(value)
        outs.append(
            None
            if kind == "allocate"
            else NumpySample(buffer, sample.sample_axis, sample.weights)
            if kind == "sample"
            else buffer
        )

    result = ufunc(*args, out=tuple(outs))

    assert isinstance(result, tuple)
    for actual, wanted, out in zip(result, expected, outs, strict=True):
        if out is not None:
            assert actual is out
        if out is None or isinstance(out, NumpySample):
            assert isinstance(actual, NumpySample)
            assert actual.sample_axis == sample.sample_axis
            np.testing.assert_array_equal(actual.weights, sample.weights)
            assert actual.dtype == wanted.dtype
        actual_array = actual.array if isinstance(actual, NumpySample) else actual
        np.testing.assert_array_equal(actual_array, wanted)


@pytest.mark.parametrize("ufunc", [np.modf, np.frexp, np.divmod])
def test_multiple_outputs_without_out(sample, ufunc):
    result = ufunc(sample, 2) if ufunc.nin == 2 else ufunc(sample)
    expected = ufunc(sample.array, 2) if ufunc.nin == 2 else ufunc(sample.array)

    for actual, wanted in zip(result, expected, strict=True):
        assert isinstance(actual, NumpySample)
        assert actual.sample_axis == 1
        np.testing.assert_array_equal(actual.array, wanted)
        np.testing.assert_array_equal(actual.weights, sample.weights)


@pytest.mark.parametrize("ufunc", [np.modf, np.frexp, np.divmod])
@pytest.mark.parametrize("out_mode", ["omit", "allocate", "first", "second"])
def test_multiple_outputs_preserve_nested_protected_representation(sample, ufunc, out_mode):
    array = np.stack([sample.array, sample.array + 0.25], axis=-1)
    nested = NumpySample(_NumpyVectors(array), sample.sample_axis, sample.weights)
    extra = (2,) if ufunc.nin == 2 else ()
    expected = ufunc(array, *extra)
    outs = [None, None]
    if out_mode in ("first", "second"):
        index = 0 if out_mode == "first" else 1
        outs[index] = NumpySample(_NumpyVectors(np.empty_like(expected[index])), sample.sample_axis, sample.weights)

    result = ufunc(nested, *extra) if out_mode == "omit" else ufunc(nested, *extra, out=tuple(outs))

    assert isinstance(result, tuple)
    for actual, wanted, out in zip(result, expected, outs, strict=True):
        if out is not None:
            assert actual is out
        assert isinstance(actual, NumpySample)
        assert isinstance(actual.array, _NumpyVectors)
        assert actual.sample_axis == sample.sample_axis
        assert actual.shape == sample.shape
        assert actual.array.protected_shape == (2,)
        assert actual.dtype == wanted.dtype
        np.testing.assert_array_equal(actual.array.array, wanted)
        np.testing.assert_array_equal(actual.weights, sample.weights)


@pytest.mark.parametrize("reflected", [False, True])
def test_multiple_outputs_track_broadcast_sample_axis(sample, reflected):
    other = np.full((4, 2, 3), 2.0)
    args = (other, sample) if reflected else (sample, other)
    native_args = (other, sample.array) if reflected else (sample.array, other)

    result = np.divmod(*args)
    expected = np.divmod(*native_args)

    for actual, wanted in zip(result, expected, strict=True):
        assert isinstance(actual, NumpySample)
        assert actual.sample_axis == 2
        np.testing.assert_array_equal(actual.array, wanted)
        np.testing.assert_array_equal(actual.weights, sample.weights)


@pytest.mark.parametrize("kind", ["allocate", "array", "sample"])
def test_single_output_and_none_placeholder(sample, kind):
    buffer = np.empty_like(sample.array)
    out = None if kind == "allocate" else NumpySample(buffer, 1, sample.weights) if kind == "sample" else buffer

    # NumPy's stubs omit custom output buffers and None slots inside out tuples.
    result = cast("Any", np.sin)(sample, out=(out,))

    if out is not None:
        assert result is out
    if out is None or isinstance(out, NumpySample):
        assert isinstance(result, NumpySample)
        assert result.sample_axis == 1
        np.testing.assert_array_equal(result.weights, sample.weights)
        result = result.array
    np.testing.assert_allclose(result, np.sin(sample.array))


def test_partial_outputs_preserve_where_mask(sample):
    buffer = np.full_like(sample.array, -1)
    where = np.array([[True, False, True], [False, True, False]])

    fractional, integral = cast("Any", np.modf)(sample, out=(buffer, None), where=where)

    assert fractional is buffer
    assert isinstance(integral, NumpySample)
    np.testing.assert_array_equal(fractional[~where], -1)
    expected_fractional, expected_integral = np.modf(sample.array)
    np.testing.assert_allclose(fractional[where], expected_fractional[where])
    np.testing.assert_allclose(integral.array[where], expected_integral[where])
    np.testing.assert_array_equal(integral.weights, sample.weights)


@pytest.mark.parametrize("sample_axis", [0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_reduction_with_allocated_output_tracks_sample_axis(sample, sample_axis, keepdims):
    result = cast("Any", np.add.reduce)(sample, axis=sample_axis, keepdims=keepdims, out=(None,))
    expected = np.add.reduce(sample.array, axis=sample_axis, keepdims=keepdims)

    if sample_axis == sample.sample_axis:
        assert not isinstance(result, NumpySample)
        np.testing.assert_allclose(result, expected)
    else:
        assert isinstance(result, NumpySample)
        assert result.sample_axis == (1 if keepdims else 0)
        np.testing.assert_allclose(result.array, expected)
        np.testing.assert_array_equal(result.weights, sample.weights)


@pytest.mark.parametrize("keepdims", [False, True])
def test_reduction_preserves_native_out_identity(sample, keepdims):
    expected = np.add.reduce(sample.array, axis=1, keepdims=keepdims)
    out = np.empty_like(expected)

    result = np.add.reduce(sample, axis=1, keepdims=keepdims, out=out)

    assert result is out
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_accumulation_preserves_sample_axis_and_weights(sample, axis):
    result = np.add.accumulate(sample, axis=axis)

    assert isinstance(result, NumpySample)
    assert result.sample_axis == sample.sample_axis
    np.testing.assert_allclose(result.array, np.add.accumulate(sample.array, axis=axis))
    np.testing.assert_array_equal(result.weights, sample.weights)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_reduceat_only_drops_wrapper_when_reducing_samples(sample, axis):
    result = np.add.reduceat(sample, [0], axis=axis)
    expected = np.add.reduceat(sample.array, [0], axis=axis)

    if axis % sample.ndim == sample.sample_axis:
        assert not isinstance(result, NumpySample)
        np.testing.assert_allclose(result, expected)
    else:
        assert isinstance(result, NumpySample)
        assert result.sample_axis == sample.sample_axis
        np.testing.assert_allclose(result.array, expected)
        np.testing.assert_array_equal(result.weights, sample.weights)


@pytest.mark.parametrize("provided_index", [None, 0, 1])
def test_multiple_outer_outputs_remain_native_and_preserve_buffers(sample, provided_index):
    other = np.array([2.0, 3.0])
    # NumPy's stubs mark outer as unsupported for two-output ufuncs.
    outer = cast("Any", np.divmod.outer)
    expected = outer(sample.array, other)
    outs = [None, None]
    if provided_index is not None:
        outs[provided_index] = np.empty_like(expected[provided_index])

    result = outer(sample, other, out=tuple(outs))

    assert isinstance(result, tuple)
    for actual, wanted, out in zip(result, expected, outs, strict=True):
        if out is not None:
            assert actual is out
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, wanted)
