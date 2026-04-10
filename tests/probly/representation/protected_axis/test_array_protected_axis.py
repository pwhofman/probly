"""Tests for generalized numpy protected-axis behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from probly.representation._protected_axis.array import ArrayAxisProtected


@dataclass(frozen=True, slots=True)
class SingleArray(ArrayAxisProtected[np.ndarray]):
    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}


@dataclass(frozen=True, slots=True)
class PairArray(ArrayAxisProtected[np.ndarray]):
    first: np.ndarray
    second: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"first": 0, "second": 0}


@dataclass(frozen=True, slots=True)
class InnerPair(ArrayAxisProtected[np.ndarray]):
    left: np.ndarray
    right: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}


@dataclass(frozen=True, slots=True)
class OuterNested(ArrayAxisProtected[Any]):
    inner: InnerPair
    aux: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "aux": 1}


def test_single_field_array_and_ufunc() -> None:
    x = SingleArray(np.arange(6.0).reshape(2, 3))

    as_array = np.asarray(x)
    np.testing.assert_array_equal(as_array, x.array)

    shifted = np.add(x, 1.0)
    assert isinstance(shifted, SingleArray)
    np.testing.assert_array_equal(shifted.array, x.array + 1.0)


def test_multi_field_metadata_comes_from_first_field() -> None:
    x = PairArray(
        first=np.ones((2, 3), dtype=np.float64),
        second=np.ones((2, 3), dtype=np.float32),
    )

    assert x.shape == (2, 3)
    assert x.ndim == 2
    assert x.size == 6
    assert x.dtype == np.float64


def test_multi_field_array_and_ufunc_raise_by_default() -> None:
    x = PairArray(np.ones((2, 2)), np.ones((2, 2)))

    with pytest.raises(TypeError, match="Cannot convert multi-field"):
        _ = np.asarray(x)

    with pytest.raises(TypeError, match="__array_ufunc__ is undefined"):
        _ = np.add(x, 1)


def test_shape_functions_apply_to_all_fields() -> None:
    x = PairArray(np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3) + 10)

    expanded = np.expand_dims(x, axis=0)
    assert isinstance(expanded, PairArray)
    assert expanded.first.shape == (1, 2, 3)
    assert expanded.second.shape == (1, 2, 3)

    reshaped = np.reshape(x, (3, 2))
    assert isinstance(reshaped, PairArray)
    assert reshaped.first.shape == (3, 2)
    assert reshaped.second.shape == (3, 2)


def test_getitem_and_setitem_for_instance_and_tuple() -> None:
    x = PairArray(np.zeros((2, 2)), np.ones((2, 2)))

    y = x[0]
    assert isinstance(y, PairArray)
    assert y.first.shape == (2,)
    assert y.second.shape == (2,)

    x[0] = PairArray(np.array([5.0, 6.0]), np.array([7.0, 8.0]))
    np.testing.assert_array_equal(x.first[0], np.array([5.0, 6.0]))
    np.testing.assert_array_equal(x.second[0], np.array([7.0, 8.0]))

    x[1] = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(x.first[1], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(x.second[1], np.array([3.0, 4.0]))


def test_nested_multi_field_value_delegates_without_array_casts() -> None:
    inner = InnerPair(
        left=np.arange(24.0).reshape(2, 3, 4),
        right=np.arange(24.0).reshape(2, 3, 4) + 100,
    )
    outer = OuterNested(inner=inner, aux=np.arange(10.0).reshape(2, 5))

    values = outer.protected_values()
    assert values["inner"] is inner

    expanded = np.expand_dims(outer, axis=0)
    assert isinstance(expanded.inner, InnerPair)
    assert expanded.inner.left.shape == (1, 2, 3, 4)
    assert expanded.inner.right.shape == (1, 2, 3, 4)
    assert expanded.aux.shape == (1, 2, 5)

    stacked = np.stack((outer, outer), axis=0)
    assert isinstance(stacked.inner, InnerPair)
    assert stacked.inner.left.shape == (2, 2, 3, 4)
    assert stacked.inner.right.shape == (2, 2, 3, 4)
    assert stacked.aux.shape == (2, 2, 5)

    sliced = outer[1]
    assert isinstance(sliced, OuterNested)
    assert isinstance(sliced.inner, InnerPair)
    assert sliced.inner.left.shape == (3, 4)
    assert sliced.inner.right.shape == (3, 4)
    assert sliced.aux.shape == (5,)
