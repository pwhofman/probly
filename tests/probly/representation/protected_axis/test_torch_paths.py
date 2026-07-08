"""Tests for ``TorchAxisProtected`` indexing, conversion, and dispatch paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation._protected_axis.torch import TorchAxisProtected


@dataclass(frozen=True, slots=True)
class _SingleTensor(TorchAxisProtected[Any]):
    """Single torch field with one protected trailing axis."""

    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}
    permitted_functions: ClassVar[set[Any]] = {torch.sum, torch.mean}


@dataclass(frozen=True, slots=True)
class _ScalarTensor(TorchAxisProtected[Any]):
    """Single torch field with no protected trailing axis."""

    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 0}


@dataclass(frozen=True, slots=True)
class _PairTensor(TorchAxisProtected[Any]):
    """Two torch fields with no protected trailing axes."""

    first: torch.Tensor
    second: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"first": 0, "second": 0}


@dataclass(frozen=True, slots=True)
class _PairTensorProtected(TorchAxisProtected[Any]):
    """Two torch fields each with one protected trailing axis."""

    left: torch.Tensor
    right: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}


@dataclass(frozen=True, slots=True)
class _NumpyOnly(TorchAxisProtected[Any]):
    """Edge case: a torch-protected representation whose only field is numpy.

    The representation is unusual but allowed; it triggers the ``_torch_protected_value``
    error path because no torch-like value is present.
    """

    sidecar: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"sidecar": 1}


# ---------------------------------------------------------------------------
# protected_values: permitted vs unpermitted func.
# ---------------------------------------------------------------------------


def test_torch_protected_values_no_args_returns_all_values() -> None:
    """``protected_values()`` without arguments returns all fields."""
    x = _SingleTensor(torch.arange(6.0).reshape(2, 3))
    values = x.protected_values()
    assert "tensor" in values
    torch.testing.assert_close(values["tensor"], x.tensor)


def test_torch_protected_values_with_permitted_func() -> None:
    """``protected_values`` with a permitted function returns the dict."""
    x = _SingleTensor(torch.arange(6.0).reshape(2, 3))
    values = x.protected_values(torch.sum)
    assert values is not None


def test_torch_protected_values_with_unpermitted_func_returns_none() -> None:
    """``protected_values`` with an unpermitted function returns None."""
    x = _SingleTensor(torch.arange(6.0).reshape(2, 3))
    assert x.protected_values(torch.cumsum) is None


# ---------------------------------------------------------------------------
# _torch_protected_value: must find at least one tensor field.
# ---------------------------------------------------------------------------


def test_torch_protected_value_raises_for_numpy_only_layout() -> None:
    """A torch-protected layout with only numpy fields raises on ``_torch_protected_value``."""
    x = _NumpyOnly(np.arange(6.0).reshape(2, 3))

    with pytest.raises(TypeError, match="No torch-like protected value"):
        _ = x._torch_protected_value()  # noqa: SLF001


# ---------------------------------------------------------------------------
# __len__ / __iter__ for zero-dim representations.
# ---------------------------------------------------------------------------


def test_len_raises_for_zero_dim_distribution() -> None:
    """``__len__`` raises on zero-dim torch protected representations."""
    x = _SingleTensor(torch.tensor([1.0, 2.0, 3.0]))
    assert x.shape == ()
    assert x.ndim == 0

    with pytest.raises(TypeError, match="unsized distribution"):
        _ = len(x)


def test_iter_yields_objects() -> None:
    """Iteration walks the leading batch axis."""
    x = _SingleTensor(torch.arange(12.0).reshape(3, 4))
    items = list(x)
    assert len(items) == 3
    for i, item in enumerate(items):
        assert isinstance(item, _SingleTensor)
        torch.testing.assert_close(item.tensor, x.tensor[i])


# ---------------------------------------------------------------------------
# __array_namespace__, dtype, device.
# ---------------------------------------------------------------------------


def test_array_namespace_delegates_to_underlying_tensor() -> None:
    """``__array_namespace__`` delegates to the underlying tensor's implementation.

    Torch tensors do not implement ``__array_namespace__`` directly, so we only
    assert that the call reaches the delegated path (raising ``AttributeError``).
    """
    x = _SingleTensor(torch.arange(6.0).reshape(2, 3))
    with pytest.raises(AttributeError):
        _ = x.__array_namespace__()


def test_dtype_property_delegates_to_torch_value() -> None:
    """``dtype`` returns the underlying torch dtype."""
    x = _SingleTensor(torch.ones((2, 3), dtype=torch.float64))
    assert x.dtype == torch.float64


def test_device_property_delegates_to_torch_value() -> None:
    """``device`` returns the underlying torch device."""
    x = _SingleTensor(torch.ones((2, 3)))
    assert x.device.type == "cpu"


# ---------------------------------------------------------------------------
# size with int dim.
# ---------------------------------------------------------------------------


def test_size_int_dim_returns_batch_size() -> None:
    """``size(int)`` returns the batch size at that dim."""
    x = _SingleTensor(torch.arange(24.0).reshape(2, 3, 4))
    assert x.size(0) == 2
    assert x.size(1) == 3
    assert x.size(-1) == 3
    assert x.size() == torch.Size((2, 3))


def test_size_int_dim_out_of_bounds_raises() -> None:
    """``size(dim)`` with out-of-range dim raises IndexError."""
    x = _SingleTensor(torch.arange(24.0).reshape(2, 3, 4))
    with pytest.raises(IndexError, match="out of bounds"):
        _ = x.size(5)


# ---------------------------------------------------------------------------
# mT and mH on insufficient ndim.
# ---------------------------------------------------------------------------


def test_mT_requires_two_batch_dims() -> None:  # noqa: N802
    """``mT`` requires ndim >= 2."""
    x = _SingleTensor(torch.arange(8.0).reshape(2, 4))  # batch ndim = 1
    with pytest.raises(ValueError, match="at least 2 batch dimensions"):
        _ = x.mT


def test_mH_requires_two_batch_dims() -> None:  # noqa: N802
    """``mH`` requires ndim >= 2."""
    x = _SingleTensor(torch.arange(8.0).reshape(2, 4))  # batch ndim = 1
    with pytest.raises(ValueError, match="at least 2 batch dimensions"):
        _ = x.mH


# ---------------------------------------------------------------------------
# Indexing patterns.
# ---------------------------------------------------------------------------


def test_getitem_int() -> None:
    """Integer indexing into batch axis works."""
    x = _SingleTensor(torch.arange(24.0).reshape(2, 3, 4))
    y = x[0]
    assert isinstance(y, _SingleTensor)
    assert tuple(y.tensor.shape) == (3, 4)


def test_getitem_slice() -> None:
    """Slice indexing into batch axis works."""
    x = _SingleTensor(torch.arange(24.0).reshape(2, 3, 4))
    y = x[:1]
    assert isinstance(y, _SingleTensor)
    assert tuple(y.tensor.shape) == (1, 3, 4)


def test_getitem_ellipsis() -> None:
    """Ellipsis indexing returns the same shape representation."""
    x = _SingleTensor(torch.arange(24.0).reshape(2, 3, 4))
    y = x[...]
    assert isinstance(y, _SingleTensor)
    assert tuple(y.tensor.shape) == (2, 3, 4)


def test_getitem_tuple() -> None:
    """Tuple indexing across batch axes works."""
    x = _SingleTensor(torch.arange(24.0).reshape(2, 3, 4))
    y = x[0, 1]
    assert isinstance(y, _SingleTensor)
    assert tuple(y.tensor.shape) == (4,)


def test_getitem_axes_zero_promotes_scalar() -> None:
    """For ``axes=0``, scalar results are promoted (torch tensor)."""
    x = _ScalarTensor(torch.arange(6.0))
    y = x[0]
    assert isinstance(y, _ScalarTensor)
    assert y.tensor.ndim == 0


# ---------------------------------------------------------------------------
# Setitem error paths.
# ---------------------------------------------------------------------------


def test_setitem_rejects_tuple_with_wrong_length() -> None:
    """Assigning a tuple with the wrong number of fields raises TypeError."""
    x = _PairTensor(torch.zeros((2, 2)), torch.ones((2, 2)))
    with pytest.raises(TypeError, match="Expected tuple"):
        x[0] = (torch.zeros(2),)


def test_setitem_rejects_scalar_for_multi_field() -> None:
    """Single value assignment is rejected for multi-field protected objects."""
    x = _PairTensor(torch.zeros((2, 2)), torch.ones((2, 2)))
    with pytest.raises(TypeError, match="multi-field protected object"):
        x[0] = torch.zeros(2)


def test_setitem_rejects_value_with_wrong_protected_shape() -> None:
    """Assigning a value whose protected trailing axes differ raises ValueError."""
    x = _SingleTensor(torch.zeros((2, 4)))
    bad = _SingleTensor(torch.zeros((1, 5)))  # protected size 5

    with pytest.raises(ValueError, match="modifies protected trailing axes"):
        x[0] = bad


# ---------------------------------------------------------------------------
# to and detach.
# ---------------------------------------------------------------------------


def test_to_returns_self_when_already_correct() -> None:
    """``to`` returns the original instance when no conversion needed."""
    x = _SingleTensor(torch.arange(6.0).reshape(2, 3))
    y = x.to(dtype=torch.float32)  # tensor already float32
    assert y is x


def test_to_changes_dtype() -> None:
    """``to`` returns a new instance with converted dtype."""
    x = _SingleTensor(torch.ones((2, 3), dtype=torch.float32))
    y = x.to(dtype=torch.float64)
    assert isinstance(y, _SingleTensor)
    assert y.tensor.dtype == torch.float64


def test_detach_returns_grad_free_copy() -> None:
    """``detach`` removes ``requires_grad`` from tensor fields."""
    tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x = _ScalarTensor(tensor)
    y = x.detach()
    assert isinstance(y, _ScalarTensor)
    assert not y.tensor.requires_grad


# ---------------------------------------------------------------------------
# numpy and __array__.
# ---------------------------------------------------------------------------


def test_numpy_returns_array_for_single_field() -> None:
    """``numpy()`` on single-field returns a numpy array."""
    x = _SingleTensor(torch.arange(6.0).reshape(2, 3))
    arr = x.numpy()
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, x.tensor.numpy())


def test_numpy_force_copies_data() -> None:
    """``numpy(force=True)`` copies the underlying buffer."""
    x = _SingleTensor(torch.arange(6.0).reshape(2, 3))
    arr = x.numpy(force=True)
    assert isinstance(arr, np.ndarray)


def test_numpy_with_numpy_field_force_returns_copy() -> None:
    """Numpy fields are returned as a copy when ``force=True``."""
    x = _NumpyOnly(np.arange(6.0).reshape(2, 3))
    arr = x.numpy(force=True)
    assert isinstance(arr, np.ndarray)
    assert arr is not x.sidecar


def test_numpy_with_numpy_field_no_force_returns_same() -> None:
    """Numpy fields are returned as-is when ``force=False``."""
    x = _NumpyOnly(np.arange(6.0).reshape(2, 3))
    arr = x.numpy(force=False)
    assert arr is x.sidecar


def test_numpy_rejects_multi_field() -> None:
    """``numpy()`` is not defined for multi-field representations."""
    x = _PairTensor(torch.zeros((2, 3)), torch.ones((2, 3)))
    with pytest.raises(TypeError, match="multi-field"):
        _ = x.numpy()


def test_array_dunder_returns_ndarray() -> None:
    """``np.asarray`` returns an ndarray with proper dtype."""
    x = _SingleTensor(torch.arange(6.0).reshape(2, 3))
    arr = np.asarray(x)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, x.tensor.numpy())


def test_array_dunder_with_dtype_converts() -> None:
    """``np.asarray(..., dtype=...)`` converts the dtype."""
    x = _SingleTensor(torch.arange(6.0, dtype=torch.float32).reshape(2, 3))
    arr = np.asarray(x, dtype=np.float64)
    assert arr.dtype == np.float64


def test_array_dunder_with_copy_true_returns_copy() -> None:
    """``np.asarray(..., copy=True)`` returns a freshly allocated array."""
    x = _SingleTensor(torch.arange(6.0).reshape(2, 3))
    arr = np.array(x, copy=True)
    assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# reshape and torch_function dispatch.
# ---------------------------------------------------------------------------


def test_reshape_method_uses_torch_function() -> None:
    """``self.reshape(...)`` is dispatched via torch_function."""
    x = _SingleTensor(torch.arange(24.0).reshape(2, 3, 4))
    y = x.reshape(6)
    assert isinstance(y, _SingleTensor)
    assert tuple(y.tensor.shape) == (6, 4)


def test_torch_function_classmethod_dispatches() -> None:
    """The class-level ``__torch_function__`` is the entry point for torch ops."""
    x = _SingleTensor(torch.arange(24.0).reshape(2, 3, 4))
    y = torch.transpose(x, 0, 1)
    assert isinstance(y, _SingleTensor)
    assert tuple(y.tensor.shape) == (3, 2, 4)
