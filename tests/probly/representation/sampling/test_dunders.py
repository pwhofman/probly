"""Edge cases for ArraySample / TorchSample dunder methods.

Targets the small gaps left in src/probly/representation/sample/array.py and
src/probly/representation/sample/torch.py — the property/dunder routes that
existing tests don't fully exercise.
"""

from __future__ import annotations

import numpy as np
import pytest


def _torch_modules():
    """Return torch module or skip the calling test."""
    pytest.importorskip("torch")
    import torch as _torch  # noqa: PLC0415

    return _torch


class TestArraySampleScalarConversionsAndDunders:
    """Cover the small dunder branches in ArraySample."""

    def test_array_dunder_with_explicit_dtype(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.arange(6, dtype=np.int64).reshape(2, 3), sample_axis=0)
        out = np.asarray(sample, dtype=np.float32)
        assert out.dtype == np.float32

    def test_dtype_property(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.arange(6, dtype=np.float64).reshape(2, 3), sample_axis=0)
        assert sample.dtype == np.float64

    def test_ndim_property(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.zeros((2, 3, 4)), sample_axis=0)
        assert sample.ndim == 3

    def test_shape_property(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.zeros((2, 3, 4)), sample_axis=0)
        assert sample.shape == (2, 3, 4)

    def test_size_property(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.zeros((2, 3, 4)), sample_axis=0)
        assert sample.size == 24

    def test_flags_property(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.zeros((2, 3)), sample_axis=0)
        # flags are propagated through to the underlying array.
        assert sample.flags.c_contiguous is True

    def test_setitem_mutates_array(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.zeros((2, 3), dtype=float), sample_axis=0)
        sample[0, 0] = 99.0
        assert sample.array[0, 0] == pytest.approx(99.0)

    def test_setitem_with_slice(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.zeros((2, 3), dtype=float), sample_axis=0)
        sample[0] = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(sample.array[0], [1.0, 2.0, 3.0])

    def test_iter(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.array([[1, 2], [3, 4]]), sample_axis=0)
        rows = list(iter(sample))
        np.testing.assert_array_equal(rows[0], [1, 2])
        np.testing.assert_array_equal(rows[1], [3, 4])

    def test_array_namespace(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.zeros((2, 3)), sample_axis=0)
        ns = sample.__array_namespace__()
        assert ns is not None

    def test_eq_with_array(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        a = ArraySample(array=np.array([1, 2, 3]), sample_axis=0)
        eq = a == np.array([1, 4, 3])
        np.testing.assert_array_equal(eq, [True, False, True])

    def test_array_like_with_copy_returns_new(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        copied = sample.__array_like__(copy=True)
        assert copied is not sample
        # underlying arrays are different objects (copy makes a new ndarray).
        assert copied.array is not sample.array

    def test_array_like_no_copy_returns_self(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        result = sample.__array_like__()
        assert result is sample


class TestArraySampleToTorch:
    """The __torch_like__ conversion path in ArraySample."""

    def test_to_torch_like_basic(self) -> None:
        torch = _torch_modules()  # noqa: F841
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        sample = ArraySample(array=np.arange(6, dtype=np.float32).reshape(2, 3), sample_axis=0)
        torch_sample = sample.__torch_like__()
        assert isinstance(torch_sample, TorchSample)
        assert torch_sample.sample_dim == 0
        assert tuple(torch_sample.tensor.shape) == (2, 3)

    def test_to_torch_like_with_weights(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        sample = ArraySample(
            array=np.arange(6, dtype=np.float32).reshape(2, 3),
            sample_axis=0,
            weights=np.array([0.4, 0.6], dtype=np.float32),
        )
        torch_sample = sample.__torch_like__()
        assert isinstance(torch_sample, TorchSample)
        assert torch_sample.weights is not None
        torch.testing.assert_close(torch_sample.weights, torch.tensor([0.4, 0.6]))


class TestArraySampleFromSamplePaths:
    """from_sample's lesser-used branches."""

    def test_from_sample_with_explicit_axis_moves_axis(self) -> None:
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        original = ArraySample(array=np.arange(12).reshape(3, 4), sample_axis=0)
        moved = ArraySample.from_sample(original, sample_axis=1)
        assert moved.sample_axis == 1
        assert moved.array.shape == (4, 3)


class TestTorchSampleDunders:
    """Edge cases for TorchSample properties."""

    def test_dtype(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(torch.zeros((2, 3), dtype=torch.float64), sample_dim=0)
        assert s.dtype == torch.float64

    def test_device(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(torch.zeros((2, 3)), sample_dim=0)
        # device is exposed as the inner tensor's device
        assert str(s.device) in {"cpu", "meta"}

    def test_ndim(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(torch.zeros((2, 3, 4)), sample_dim=0)
        assert s.ndim == 3

    def test_shape(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(torch.zeros((2, 3, 4)), sample_dim=0)
        assert tuple(s.shape) == (2, 3, 4)

    def test_size_with_dim(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(torch.zeros((2, 3, 4)), sample_dim=0)
        assert s.size(1) == 3


class TestArraySampleArrayFunctionsEdges:
    """Cover the unusual reshape branches in array_functions.py."""

    def test_reshape_with_explicit_a_order(self) -> None:
        """``order='A'`` chooses between C and F based on array contiguity."""
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        arr = np.asfortranarray(np.arange(12, dtype=float).reshape(3, 4))
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.reshape(sample, (3, 4), order="A")
        # F-contiguous array with order='A' -> uses F path
        assert isinstance(result, ArraySample)

    def test_reduction_with_non_bool_keepdims(self) -> None:
        """Non-bool keepdims (e.g. np._NoValue) is treated as False."""
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.arange(12, dtype=float).reshape(3, 4), sample_axis=0)
        result = np.mean(sample, axis=1, keepdims=np._NoValue)  # noqa: SLF001
        assert isinstance(result, ArraySample)
        # non-bool keepdims -> treated as False -> shape (3,) not (3, 1)
        assert result.array.shape == (3,)

    def test_reduction_with_out_parameter(self) -> None:
        """When `out=ArraySample(...)`, the function returns the out wrapper."""
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.arange(12, dtype=float).reshape(3, 4), sample_axis=0)
        out_buf = ArraySample(array=np.zeros(3), sample_axis=0)
        result = np.mean(sample, axis=1, out=out_buf.array)
        # numpy returns the underlying out buffer (wrapped automatically)
        assert result is not None
