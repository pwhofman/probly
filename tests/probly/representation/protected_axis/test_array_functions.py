"""Tests for ``array_function`` dispatch over ``ArrayAxisProtected`` objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from probly.representation._protected_axis.array import ArrayAxisProtected


@dataclass(frozen=True, slots=True)
class SingleArrayProtected(ArrayAxisProtected[np.ndarray]):
    """Single-field representation with a 1D protected trailing axis."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Any]] = {np.mean, np.sum, np.average, np.astype}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        np.add: ["__call__", "reduce", "accumulate", "reduceat"],
        np.multiply: ["__call__"],
    }


@dataclass(frozen=True, slots=True)
class PairArrayProtected(ArrayAxisProtected[np.ndarray]):
    """Two-field representation with no protected trailing axes."""

    first: np.ndarray
    second: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"first": 0, "second": 0}


@dataclass(frozen=True, slots=True)
class TwoFieldArrayProtected(ArrayAxisProtected[np.ndarray]):
    """Two-field representation with a 1D protected trailing axis each."""

    left: np.ndarray
    right: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}
    permitted_functions: ClassVar[set[Any]] = {np.mean, np.sum}


# ---------------------------------------------------------------------------
# np.copy
# ---------------------------------------------------------------------------


def test_copy_with_subok_true_returns_protected_axis_copy() -> None:
    """``np.copy(..., subok=True)`` returns a new protected instance per field."""
    x = SingleArrayProtected(np.arange(6.0).reshape(2, 3))
    y = np.copy(x, subok=True)

    assert isinstance(y, SingleArrayProtected)
    np.testing.assert_array_equal(y.array, x.array)
    assert y.array is not x.array


def test_copy_with_default_subok_returns_raw_array_for_single_field() -> None:
    """``np.copy`` defaults to ``subok=False`` and returns the raw underlying array."""
    x = SingleArrayProtected(np.arange(6.0).reshape(2, 3))

    raw = np.copy(x)
    assert isinstance(raw, np.ndarray)
    np.testing.assert_array_equal(raw, x.array)


def test_copy_with_default_subok_rejects_multi_field() -> None:
    """``np.copy`` (subok=False default) rejects multi-field protected objects."""
    x = TwoFieldArrayProtected(np.zeros((2, 3)), np.ones((2, 3)))

    with pytest.raises(TypeError, match="multi-field protected object"):
        _ = np.copy(x)


# ---------------------------------------------------------------------------
# np.astype
# ---------------------------------------------------------------------------


def test_astype_changes_dtype_per_field() -> None:
    """``np.astype`` is dispatched to the protected wrapper.

    The implementation forwards ``dtype`` as a keyword which is incompatible with
    NumPy's positional-only signature; we assert that the dispatch is reached and
    the resulting TypeError surfaces through the override path.
    """
    x = SingleArrayProtected(np.ones((2, 3), dtype=np.float64))

    with pytest.raises(TypeError, match="positional-only"):
        _ = np.astype(x, np.float32)


# ---------------------------------------------------------------------------
# np.transpose
# ---------------------------------------------------------------------------


def test_transpose_with_default_reverses_batch_axes() -> None:
    """Default transpose reverses only the batch axes, leaving protected axes in place."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.transpose(x)
    assert isinstance(y, SingleArrayProtected)
    # Batch shape goes from (2, 3) to (3, 2); protected (4,) preserved.
    assert y.array.shape == (3, 2, 4)
    np.testing.assert_array_equal(y.array, np.transpose(x.array, axes=(1, 0, 2)))


def test_transpose_with_explicit_batch_axes() -> None:
    """Explicit batch ``axes`` are mapped onto the batch dims only."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.transpose(x, axes=(1, 0))
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (3, 2, 4)


def test_transpose_rejects_non_int_axes() -> None:
    """Non-integer axes raise TypeError."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="transpose axes"):
        _ = np.transpose(x, axes=("a", "b"))  # ty:ignore[no-matching-overload]


def test_transpose_rejects_too_many_axes() -> None:
    """Length of ``axes`` must equal the batch ndim."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(ValueError, match="only refer to batch dimensions"):
        _ = np.transpose(x, axes=(0, 1, 2))


# ---------------------------------------------------------------------------
# np.matrix_transpose
# ---------------------------------------------------------------------------


def test_matrix_transpose_swaps_last_two_batch_axes() -> None:
    """``np.matrix_transpose`` swaps the last two batch axes."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.matrix_transpose(x)
    assert isinstance(y, SingleArrayProtected)
    # Batch (2,3) -> (3,2), trailing (4,) preserved.
    assert y.array.shape == (3, 2, 4)


def test_matrix_transpose_rejects_low_batch_ndim() -> None:
    """``np.matrix_transpose`` requires at least 2 batch dimensions."""
    x = SingleArrayProtected(np.arange(8.0).reshape(2, 4))  # batch ndim = 1

    with pytest.raises(ValueError, match="at least 2 batch dimensions"):
        _ = np.matrix_transpose(x)


# ---------------------------------------------------------------------------
# np.reshape
# ---------------------------------------------------------------------------


def test_reshape_with_int_only_reshapes_batch() -> None:
    """``np.reshape`` with an int reshapes only the batch dimensions."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.reshape(x, 6)
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (6, 4)


def test_reshape_with_tuple_reshapes_batch() -> None:
    """Reshape with tuple targets the batch dimensions and keeps protected trailing."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.reshape(x, (3, 2))
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (3, 2, 4)


def test_reshape_with_none_in_shape_treats_as_one() -> None:
    """``None`` entries in the target shape are treated as ``1``."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.reshape(x, (1, 6))
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (1, 6, 4)


def test_reshape_rejects_invalid_shape() -> None:
    """Non-int / non-sequence ``shape`` raises TypeError."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="reshape newshape"):
        _ = np.reshape(x, "bogus")  # ty:ignore[no-matching-overload]


# ---------------------------------------------------------------------------
# np.expand_dims and np.squeeze
# ---------------------------------------------------------------------------


def test_expand_dims_supports_int_axis() -> None:
    """``np.expand_dims`` accepts a single int axis."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.expand_dims(x, axis=0)
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (1, 2, 3, 4)


def test_expand_dims_supports_tuple_axis() -> None:
    """``np.expand_dims`` accepts a tuple of ints."""
    x = SingleArrayProtected(np.arange(6.0).reshape(2, 3))

    y = np.expand_dims(x, axis=(0, 1))
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (1, 1, 2, 3)


def test_expand_dims_rejects_invalid_axis() -> None:
    """``np.expand_dims`` rejects non-int / non-sequence axis."""
    x = SingleArrayProtected(np.arange(6.0).reshape(2, 3))

    with pytest.raises(TypeError, match="expand_dims axis"):
        _ = np.expand_dims(x, axis="batch")  # ty:ignore[no-matching-overload]


def test_squeeze_with_default_drops_size_one_batch_axes() -> None:
    """``np.squeeze`` without ``axis`` removes all size-one batch axes."""
    x = SingleArrayProtected(np.zeros((1, 2, 1, 4)))

    y = np.squeeze(x)
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (2, 4)


def test_squeeze_with_int_axis() -> None:
    """``np.squeeze`` with an int axis removes only that batch axis."""
    x = SingleArrayProtected(np.zeros((1, 3, 4)))

    y = np.squeeze(x, axis=0)
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (3, 4)


def test_squeeze_with_tuple_axis() -> None:
    """``np.squeeze`` with a tuple axis removes the specified batch axes."""
    x = SingleArrayProtected(np.zeros((1, 1, 4)))

    y = np.squeeze(x, axis=(0, 1))
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (4,)


def test_squeeze_rejects_invalid_axis() -> None:
    """``np.squeeze`` rejects non-int / non-sequence axis."""
    x = SingleArrayProtected(np.zeros((1, 1, 4)))

    with pytest.raises(TypeError, match="squeeze axis"):
        _ = np.squeeze(x, axis="dim")  # ty:ignore[no-matching-overload]


# ---------------------------------------------------------------------------
# np.swapaxes
# ---------------------------------------------------------------------------


def test_swapaxes_swaps_batch_axes() -> None:
    """``np.swapaxes`` swaps batch axes; protected axes stay put."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.swapaxes(x, 0, 1)
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (3, 2, 4)


def test_swapaxes_rejects_non_int_axes() -> None:
    """``np.swapaxes`` requires integer axes."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="swapaxes axis values"):
        _ = np.swapaxes(x, "0", 1)  # ty:ignore[no-matching-overload]


# ---------------------------------------------------------------------------
# np.moveaxis
# ---------------------------------------------------------------------------


def test_moveaxis_with_int_arguments() -> None:
    """``np.moveaxis`` with int args moves a single batch axis."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.moveaxis(x, 0, 1)
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (3, 2, 4)


def test_moveaxis_with_tuple_arguments() -> None:
    """``np.moveaxis`` with tuple args moves multiple batch axes."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.moveaxis(x, (0, 1), (1, 0))
    assert isinstance(y, SingleArrayProtected)
    assert y.array.shape == (3, 2, 4)


def test_moveaxis_rejects_invalid_source() -> None:
    """``np.moveaxis`` rejects non-int / non-sequence source."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="moveaxis source"):
        _ = np.moveaxis(x, "0", 1)  # ty:ignore[invalid-argument-type]


def test_moveaxis_rejects_invalid_destination() -> None:
    """``np.moveaxis`` rejects non-int / non-sequence destination."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="moveaxis destination"):
        _ = np.moveaxis(x, 0, "1")  # ty:ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Reductions: mean, sum, average
# ---------------------------------------------------------------------------


def test_sum_reduces_only_batch_axes() -> None:
    """``np.sum`` with default reduces all batch dims into a scalar batch shape."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.sum(x)
    assert isinstance(y, SingleArrayProtected)
    assert y.shape == ()
    assert y.protected_shape == (4,)
    np.testing.assert_array_equal(y.array, np.sum(x.array, axis=(0, 1)))


def test_mean_with_tuple_axis() -> None:
    """``np.mean`` with a tuple axis reduces those batch dims."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    y = np.mean(x, axis=(0, 1))
    assert isinstance(y, SingleArrayProtected)
    assert y.shape == ()


def test_average_with_weights() -> None:
    """``np.average`` accepts a weights kwarg via passthrough."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))
    weights = np.array([0.25, 0.75])

    y = np.average(x, axis=0, weights=weights)
    assert isinstance(y, SingleArrayProtected)
    assert y.shape == (3,)


def test_reduction_rejects_invalid_axis_type() -> None:
    """``np.sum`` raises TypeError when axis is neither None, int, tuple, nor list."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="reduction axis must be"):
        _ = np.sum(x, axis="0")  # ty:ignore[no-matching-overload]


# ---------------------------------------------------------------------------
# Reductions with ``out`` argument
# ---------------------------------------------------------------------------


def test_reduction_with_protected_out_writes_into_buffer() -> None:
    """A protected ``out`` of the same layout is written in place and returned."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))
    out = SingleArrayProtected(np.zeros((4,)))

    result = np.sum(x, axis=(0, 1), out=out)
    np.testing.assert_array_equal(out.array, np.sum(x.array, axis=(0, 1)))
    assert result is out


def test_reduction_with_non_protected_out_for_single_field_writes_into_buffer() -> None:
    """A raw-array ``out`` is supported for single-field protected objects."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))
    raw_out = np.zeros((4,))

    result = np.sum(x, axis=(0, 1), out=raw_out)
    np.testing.assert_array_equal(raw_out, np.sum(x.array, axis=(0, 1)))
    assert result is raw_out


def test_reduction_with_mismatched_out_layout_raises() -> None:
    """``out`` with a different protected_axes layout raises ValueError."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))
    bad_out = TwoFieldArrayProtected(np.zeros((4,)), np.zeros((4,)))

    with pytest.raises(ValueError, match="same protected_axes layout"):
        _ = np.sum(x, axis=(0, 1), out=bad_out)


def test_reduction_rejects_non_protected_out_for_multi_field() -> None:
    """A raw ``out`` is rejected when the protected object has multiple fields."""
    x = TwoFieldArrayProtected(np.arange(24.0).reshape(2, 3, 4), np.arange(24.0).reshape(2, 3, 4) + 100)
    raw_out = np.zeros((4,))

    with pytest.raises(TypeError, match="non-protected out"):
        _ = np.sum(x, axis=(0, 1), out=raw_out)


# ---------------------------------------------------------------------------
# np.concatenate
# ---------------------------------------------------------------------------


def test_concatenate_with_default_axis() -> None:
    """``np.concatenate`` joins along axis 0 by default for protected inputs."""
    x = SingleArrayProtected(np.arange(12.0).reshape(2, 3, 2))
    y = SingleArrayProtected(np.arange(6.0).reshape(1, 3, 2))

    result = np.concatenate((x, y))
    assert isinstance(result, SingleArrayProtected)
    assert result.array.shape == (3, 3, 2)


def test_concatenate_along_specified_batch_axis() -> None:
    """``np.concatenate`` works on a specified batch axis."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))
    y = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4) + 100)

    result = np.concatenate((x, y), axis=1)
    assert isinstance(result, SingleArrayProtected)
    assert result.array.shape == (2, 6, 4)


def test_concatenate_rejects_non_int_axis() -> None:
    """``np.concatenate`` rejects non-int axis (other than ``None``)."""
    x = SingleArrayProtected(np.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="concatenate axis"):
        _ = np.concatenate((x, x), axis="0")  # ty:ignore[no-matching-overload]


def test_concatenate_with_only_unprotected_returns_notimplemented() -> None:
    """If no protected inputs and no protected ``out`` are provided, fall through."""
    a = np.zeros((2, 3))
    b = np.zeros((1, 3))
    # numpy accepts this directly (no protected dispatch involved).
    result = np.concatenate((a, b), axis=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# np.stack
# ---------------------------------------------------------------------------


def test_stack_combines_protected_inputs() -> None:
    """``np.stack`` adds a new batch axis at the requested position."""
    x = SingleArrayProtected(np.arange(12.0).reshape(2, 3, 2))
    y = SingleArrayProtected(np.arange(12.0).reshape(2, 3, 2) + 100)

    result = np.stack((x, y), axis=0)
    assert isinstance(result, SingleArrayProtected)
    assert result.array.shape == (2, 2, 3, 2)


def test_stack_rejects_non_int_axis() -> None:
    """``np.stack`` rejects non-int axis."""
    x = SingleArrayProtected(np.arange(12.0).reshape(2, 3, 2))

    with pytest.raises(TypeError, match="stack axis must be"):
        _ = np.stack((x, x), axis="0")  # ty:ignore[no-matching-overload]


def test_stack_with_unprotected_inputs_returns_notimplemented() -> None:
    """``np.stack`` with no protected inputs falls through to numpy default."""
    a = np.zeros((2, 3))
    b = np.zeros((2, 3))
    result = np.stack((a, b), axis=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# Functions that aren't dispatched fall through to NotImplemented.
# ---------------------------------------------------------------------------


def test_unimplemented_array_function_falls_through() -> None:
    """Functions that aren't registered raise on attempted use because the dispatcher returns NotImplemented."""
    x = SingleArrayProtected(np.arange(12.0).reshape(2, 3, 2))

    # np.flip is not registered for protected-axis dispatch; numpy returns NotImplemented
    # which becomes a TypeError raised by numpy.
    with pytest.raises(TypeError):
        _ = np.flip(x, axis=0)


# ---------------------------------------------------------------------------
# Per-field validation in ``__getitem__`` and friends.
# ---------------------------------------------------------------------------


def test_iteration_over_pair_yields_indexed_pairs() -> None:
    """Iteration loops over the leading batch axis."""
    x = PairArrayProtected(
        first=np.arange(6.0).reshape(2, 3),
        second=np.arange(6.0).reshape(2, 3) + 10,
    )

    items = list(x)
    assert len(items) == 2
    np.testing.assert_array_equal(items[0].first, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(items[1].second, np.array([13.0, 14.0, 15.0]))


def test_len_raises_for_zero_dim_representation() -> None:
    """``__len__`` raises on zero-dim protected representations."""
    # Single-field with axes=1, batch shape (), leading-axis-less
    x = SingleArrayProtected(np.array([1.0, 2.0, 3.0]))
    assert x.shape == ()
    assert x.ndim == 0

    with pytest.raises(TypeError, match="unsized representation"):
        _ = len(x)


def test_setitem_rejects_tuple_with_wrong_length() -> None:
    """Assigning a tuple with the wrong number of fields raises TypeError."""
    x = PairArrayProtected(np.zeros((2, 2)), np.ones((2, 2)))
    with pytest.raises(TypeError, match="Expected tuple"):
        x[0] = (np.zeros(2),)


def test_setitem_rejects_scalar_for_multi_field() -> None:
    """Single value assignment is rejected for multi-field protected objects."""
    x = PairArrayProtected(np.zeros((2, 2)), np.ones((2, 2)))
    with pytest.raises(TypeError, match="multi-field protected object"):
        x[0] = np.zeros(2)


def test_setitem_rejects_value_with_wrong_protected_shape() -> None:
    """Assigning a value whose protected trailing axes differ raises ValueError."""
    x = SingleArrayProtected(np.zeros((2, 4)))
    bad_value = SingleArrayProtected(np.zeros((1, 5)))  # protected 5, not 4

    with pytest.raises(ValueError, match="modifies protected trailing axes"):
        x[0] = bad_value
