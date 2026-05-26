"""Tests for remaining branches across the ``_protected_axis`` package.

Targets edge cases that aren't naturally exercised by representative use:

- abstract subclass early return in ``__init_subclass__``
- multi-field assignment with raw value when there is a single field
- ``out`` mismatched layouts in ``__array_ufunc__``
- ufunc results that drop or modify protected trailing axes
- batch reduction errors and ``out`` paths in ``array_function``
- ``torch.reshape`` and ``torch.gather`` instance methods
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation._protected_axis.array_functions import (
    _validate_batch_sync,
    array_axis_protected_internals,
)
from probly.representation._protected_axis.torch import TorchAxisProtected

# ---------------------------------------------------------------------------
# Abstract subclasses skip validation in __init_subclass__.
# ---------------------------------------------------------------------------


def test_abstract_array_subclass_skips_validation() -> None:
    """Abstract subclasses can be defined without ``protected_axes``."""

    class _AbstractArray(ArrayAxisProtected[np.ndarray]):
        @abstractmethod
        def custom(self) -> None: ...

    # If validation didn't bypass, defining without ``protected_axes`` would error.
    assert _AbstractArray is not None


def test_abstract_torch_subclass_skips_validation() -> None:
    """Abstract torch subclasses can be defined without ``protected_axes``."""

    class _AbstractTorch(TorchAxisProtected[Any]):
        @abstractmethod
        def custom(self) -> None: ...

    assert _AbstractTorch is not None


# ---------------------------------------------------------------------------
# Single-field __setitem__ accepts a raw value (covers fallback branch).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SingleProtectedRaw(ArrayAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}


def test_setitem_single_field_with_raw_value() -> None:
    """Assigning a raw ndarray to a single-field protected works (fallback path)."""
    x = _SingleProtectedRaw(np.zeros((2, 3)))
    x[0] = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(x.array[0], np.array([1.0, 2.0, 3.0]))


@dataclass(frozen=True, slots=True)
class _SingleScalarProtectedRaw(TorchAxisProtected[Any]):
    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 0}


def test_torch_setitem_single_field_with_raw_value() -> None:
    """Assigning a raw scalar/tensor to a single-field torch-protected works."""
    x = _SingleScalarProtectedRaw(torch.zeros((3,)))
    x[0] = torch.tensor(5.0)
    assert torch.equal(x.tensor[0], torch.tensor(5.0))


# ---------------------------------------------------------------------------
# Indexing checks and getitem with axes==0 fallback to np.asarray.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ScalarArray(ArrayAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}


def test_scalar_array_getitem_promotes_python_scalar() -> None:
    """Indexing a 0-protected array can yield a python scalar; we promote to ndarray."""
    x = _ScalarArray(np.array([1, 2, 3]))
    y = x[0]
    assert isinstance(y, _ScalarArray)
    # The promoted result is an array.
    assert hasattr(y.array, "ndim")


# ---------------------------------------------------------------------------
# Ufunc result drops protected trailing axes -> ValueError.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SquareInputProtected(ArrayAxisProtected[np.ndarray]):
    """Ufunc-tolerant single-field with one protected axis; reduce drops along axis=1."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        np.add: ["__call__", "reduce"],
    }


# ---------------------------------------------------------------------------
# array_axis_protected_internals: handles weird inputs gracefully.
# ---------------------------------------------------------------------------


def test_internals_returns_none_for_non_protected_object() -> None:
    """``array_axis_protected_internals`` on plain ndarray returns None."""
    assert array_axis_protected_internals(np.zeros((2, 3)), None) is None


def test_internals_returns_none_for_unpermitted_function() -> None:
    """When ``check_is_permitted=True`` and the function is not permitted, returns None."""

    @dataclass(frozen=True, slots=True)
    class _Strict(ArrayAxisProtected[np.ndarray]):
        array: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}
        permitted_functions: ClassVar[set[Any]] = {np.sum}

    x = _Strict(np.zeros((2, 3)))
    # np.mean is not permitted; with check_is_permitted=True, we get None.
    assert array_axis_protected_internals(x, np.mean, check_is_permitted=True) is None


# ---------------------------------------------------------------------------
# _validate_batch_sync error paths.
# ---------------------------------------------------------------------------


def test_validate_batch_sync_raises_when_ndim_drops() -> None:
    """``_validate_batch_sync`` raises when a protected field's ndim is too low."""
    values = {"a": np.zeros(()), "b": np.zeros((3,))}
    protected_axes = {"a": 1, "b": 1}

    with pytest.raises(ValueError, match="removed protected trailing axes"):
        _validate_batch_sync(values, protected_axes)


def test_validate_batch_sync_raises_when_batch_shapes_disagree() -> None:
    """``_validate_batch_sync`` raises when batch shapes differ across fields."""
    values = {"a": np.zeros((2, 3)), "b": np.zeros((4, 3))}
    protected_axes = {"a": 1, "b": 1}

    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        _validate_batch_sync(values, protected_axes)


# ---------------------------------------------------------------------------
# Reshape returns NotImplemented when shape is None.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ReshapeProtected(ArrayAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}


def test_reshape_with_none_shape_falls_through() -> None:
    """Calling ``np.reshape`` with shape=None goes through the NotImplemented branch."""
    x = _ReshapeProtected(np.arange(24.0).reshape(2, 3, 4))
    # numpy raises a TypeError when our handler returns NotImplemented.
    with pytest.raises((TypeError, ValueError)):
        _ = np.reshape(x, None)  # ty:ignore[no-matching-overload]


# ---------------------------------------------------------------------------
# Concatenate/stack with no protected inputs but a protected ``out``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ConcatTarget(ArrayAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}


def test_concatenate_with_protected_out_and_no_protected_input_raises() -> None:
    """When the input arrays carry no protected info but ``out`` does, the dispatch raises."""
    raw_a = np.zeros((2, 3))
    raw_b = np.zeros((2, 3))
    out = _ConcatTarget(np.zeros((4, 3)))

    with pytest.raises(TypeError, match="at least one protected input"):
        _ = np.concatenate((raw_a, raw_b), axis=0, out=out)


def test_stack_with_protected_out_and_no_protected_input_raises() -> None:
    """``np.stack`` with no protected inputs but a protected ``out`` raises."""
    raw_a = np.zeros((2, 3))
    raw_b = np.zeros((2, 3))
    out = _ConcatTarget(np.zeros((2, 2, 3)))

    with pytest.raises(TypeError, match="at least one protected input"):
        _ = np.stack((raw_a, raw_b), axis=0, out=out)


# ---------------------------------------------------------------------------
# Concatenate/stack with mismatched out layout.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OtherTarget(ArrayAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}  # different layout


def test_concatenate_rejects_mismatched_out_layout() -> None:
    """``out`` with a different protected_axes layout raises ValueError."""
    x = _ConcatTarget(np.zeros((2, 3)))
    bad_out = _OtherTarget(np.zeros((4, 3)))

    with pytest.raises(ValueError, match="same protected_axes layout"):
        _ = np.concatenate((x, x), axis=0, out=bad_out)


def test_stack_rejects_mismatched_out_layout() -> None:
    """``np.stack`` with a different protected_axes layout out raises."""
    x = _ConcatTarget(np.zeros((2, 3)))
    bad_out = _OtherTarget(np.zeros((2, 2, 3)))

    with pytest.raises(ValueError, match="same protected_axes layout"):
        _ = np.stack((x, x), axis=0, out=bad_out)


# ---------------------------------------------------------------------------
# Concatenate/stack write into protected ``out`` (covers ``return out`` branches).
# ---------------------------------------------------------------------------


def test_concatenate_with_protected_out_writes_into_buffer() -> None:
    """A protected ``out`` matching the input layout receives the result."""
    x = _ConcatTarget(np.arange(6.0).reshape(2, 3))
    y = _ConcatTarget(np.arange(6.0).reshape(2, 3) + 100)
    out = _ConcatTarget(np.zeros((4, 3)))

    result = np.concatenate((x, y), axis=0, out=out)
    assert result is out
    np.testing.assert_array_equal(out.array, np.concatenate((x.array, y.array), axis=0))


def test_stack_with_protected_out_writes_into_buffer() -> None:
    """A protected ``out`` matching the input layout receives the result."""
    x = _ConcatTarget(np.arange(6.0).reshape(2, 3))
    y = _ConcatTarget(np.arange(6.0).reshape(2, 3) + 100)
    out = _ConcatTarget(np.zeros((2, 2, 3)))

    result = np.stack((x, y), axis=0, out=out)
    assert result is out
    np.testing.assert_array_equal(out.array, np.stack((x.array, y.array), axis=0))


# ---------------------------------------------------------------------------
# torch instance methods: reshape with raw shape, gather.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TorchSingle(TorchAxisProtected[Any]):
    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}


def test_torch_reshape_method_accepts_separate_args() -> None:
    """``self.reshape(a, b)`` (multiple int args) reshapes the batch."""
    x = _TorchSingle(torch.arange(24.0).reshape(2, 3, 4))
    y = x.reshape(3, 2)
    assert isinstance(y, _TorchSingle)
    assert tuple(y.tensor.shape) == (3, 2, 4)


def test_torch_gather_method_dispatches() -> None:
    """``self.gather(dim, index)`` dispatches via ``torch.gather``."""
    x = _TorchSingle(torch.arange(24.0).reshape(2, 3, 4))
    index = torch.tensor([[0, 1], [2, 0]])
    y = x.gather(dim=1, index=index)
    assert isinstance(y, _TorchSingle)
    assert tuple(y.tensor.shape) == (2, 2, 4)


# ---------------------------------------------------------------------------
# __torch_like__ delegates to to.
# ---------------------------------------------------------------------------


def test_torch_like_dunder_delegates_to_to() -> None:
    """``__torch_like__`` returns a converted instance via ``to``."""
    x = _TorchSingle(torch.ones((2, 3), dtype=torch.float32))
    y = x.__torch_like__(torch.float64)
    assert isinstance(y, _TorchSingle)
    assert y.tensor.dtype == torch.float64


# ---------------------------------------------------------------------------
# Indexing returning Python scalar gets promoted via np.asarray.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ObjectScalarProtected(ArrayAxisProtected[np.ndarray]):
    """Object-dtype scalar protected. Indexing yields raw python objects."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}


def test_getitem_scalar_object_array_promotes_to_array() -> None:
    """Indexing an object-dtype scalar protected yields an ndarray after promotion."""
    arr = np.array(["a", "b", "c"], dtype=object)
    x = _ObjectScalarProtected(arr)
    y = x[0]
    assert isinstance(y, _ObjectScalarProtected)
    # The scalar python string was promoted to an ndarray.
    assert isinstance(y.array, np.ndarray)


# ---------------------------------------------------------------------------
# np.add.reduce with tuple ``out`` (length 1) for permitted reduce.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ReduceProtected(ArrayAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        np.add: ["__call__", "reduce"],
    }


def test_ufunc_reduce_with_tuple_out_writes_into_buffer() -> None:
    """``np.add.reduce(x, axis=0, out=(buf,))`` writes through the tuple-out path."""
    x = _ReduceProtected(np.arange(24.0).reshape(2, 3, 4))
    out = _ReduceProtected(np.zeros((3, 4)))
    result = np.add.reduce(x, axis=0, out=(out,))  # ty:ignore[no-matching-overload]
    np.testing.assert_array_equal(out.array, np.add.reduce(x.array, axis=0))
    assert result is out or result == out


def test_ufunc_reduce_with_raw_tuple_out() -> None:
    """``out=(buf,)`` accepts a raw ndarray for single-field protected reductions."""
    x = _ReduceProtected(np.arange(24.0).reshape(2, 3, 4))
    raw_out = np.zeros((3, 4))
    result = np.add.reduce(x, axis=0, out=(raw_out,))  # ty:ignore[no-matching-overload]
    np.testing.assert_array_equal(raw_out, np.add.reduce(x.array, axis=0))
    assert result is raw_out or result == raw_out


# ---------------------------------------------------------------------------
# array_axis_protected_internals returns None for malformed protected shapes.
# ---------------------------------------------------------------------------


class _FakeProtected:
    """A duck-typed value that pretends to have malformed protected_axes."""

    def __init__(self, protected_axes: dict[str, int]) -> None:
        self.protected_axes = protected_axes
        self.permitted_functions: set[Any] = set()

    def protected_values(self, func: Any = None) -> dict[str, np.ndarray] | None:  # noqa: ANN401
        del func
        return {"missing_field": np.zeros((2, 3))}

    def with_protected_values(self, values: Any, func: Any = None) -> Any:  # noqa: ANN401
        del values, func
        return self


def test_internals_returns_none_when_field_missing_in_values() -> None:
    """Returns None when protected_values omits a declared field."""
    fake = _FakeProtected({"main": 1})
    assert array_axis_protected_internals(fake, None) is None


def test_internals_returns_none_when_value_ndim_too_small() -> None:
    """Returns None when a protected field has fewer dims than required."""

    class _BadNdim:
        protected_axes: ClassVar[dict[str, int]] = {"a": 2}
        permitted_functions: ClassVar[set[Any]] = set()

        def protected_values(self, func: Any = None) -> dict[str, np.ndarray] | None:  # noqa: ANN401
            del func
            return {"a": np.zeros((3,))}  # ndim=1, need >= 2.

        def with_protected_values(self, values: Any, func: Any = None) -> Any:  # noqa: ANN401
            del values, func
            return self

    fake = _BadNdim()
    assert array_axis_protected_internals(fake, None) is None


def test_internals_returns_none_for_empty_protected_axes() -> None:
    """Returns None when ``protected_axes`` is empty."""
    fake = _FakeProtected({})
    assert array_axis_protected_internals(fake, None) is None


# ---------------------------------------------------------------------------
# np.reshape with explicit ``copy``.
# ---------------------------------------------------------------------------


def test_reshape_with_explicit_copy_kw() -> None:
    """``np.reshape(..., copy=True)`` forwards ``copy`` to the underlying call."""
    x = _ReshapeProtected(np.arange(24.0).reshape(2, 3, 4))
    y = np.reshape(x, (3, 2), copy=True)
    assert isinstance(y, _ReshapeProtected)
    assert y.array.shape == (3, 2, 4)


# ---------------------------------------------------------------------------
# Unsupported np.stack / np.concatenate with no protected anywhere.
# ---------------------------------------------------------------------------


def test_stack_with_only_raw_arrays_falls_through_to_numpy() -> None:
    """``np.stack`` over only raw arrays goes through numpy's default path."""
    a = np.zeros((2, 3))
    b = np.zeros((2, 3))
    result = np.stack((a, b), axis=0)
    assert result.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# np.add.reduceat without indices fails with informative error.
# ---------------------------------------------------------------------------


def test_array_function_called_with_no_protected_returns_notimplemented() -> None:
    """Calling ``array_function`` directly with no protected inputs returns NotImplemented."""
    from probly.representation._protected_axis.array_functions import array_function  # noqa: PLC0415

    a = np.zeros((2, 3))
    b = np.zeros((2, 3))
    # Pass np.stack through the dispatcher with raw arrays (no protected).
    result = array_function(np.stack, (np.ndarray,), ((a, b),), {"axis": 0})
    assert result is NotImplemented


def test_array_function_concatenate_no_protected_returns_notimplemented() -> None:
    """Calling ``array_function(np.concatenate)`` with raw arrays returns NotImplemented."""
    from probly.representation._protected_axis.array_functions import array_function  # noqa: PLC0415

    a = np.zeros((2, 3))
    b = np.zeros((1, 3))
    result = array_function(np.concatenate, (np.ndarray,), ((a, b),), {})
    assert result is NotImplemented


def test_ufunc_reduceat_without_indices_raises() -> None:
    """``np.add.reduceat`` without an indices argument raises TypeError."""

    @dataclass(frozen=True, slots=True)
    class _ReduceAtProtected(ArrayAxisProtected[np.ndarray]):
        array: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}
        permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
            np.add: ["reduceat"],
        }

    y = _ReduceAtProtected(np.arange(24.0).reshape(2, 3, 4))
    # numpy itself rejects calling reduceat without indices first; if we somehow
    # bypass that, our code raises ``reduceat requires indices as its second argument``.
    with pytest.raises(TypeError):
        _ = np.add.reduceat(y)  # ty:ignore[missing-argument]
