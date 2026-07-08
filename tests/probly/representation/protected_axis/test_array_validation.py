"""Tests for ``ArrayAxisProtected`` ``__init_subclass__`` validation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from probly.representation._protected_axis.array import (
    ArrayAxisProtected,
    _validate_field_ndim,
)


def test_validate_field_ndim_raises_when_too_low() -> None:
    """Helper raises ValueError when ndim is less than required protected axes."""
    with pytest.raises(ValueError, match="ndim 1, expected >= 2"):
        _validate_field_ndim("field", 1, 2)


def test_validate_field_ndim_passes_when_sufficient() -> None:
    """Helper returns silently when ndim is sufficient."""
    _validate_field_ndim("field", 2, 2)
    _validate_field_ndim("field", 5, 2)


def test_subclass_requires_non_empty_protected_axes() -> None:
    """Subclasses must define ``protected_axes`` as a non-empty dict."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _Empty(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {}


def test_subclass_rejects_non_dict_protected_axes() -> None:
    """Subclasses with non-dict ``protected_axes`` raise TypeError."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _BadAxes(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[Any] = ["array"]


def test_subclass_rejects_empty_string_field_name() -> None:
    """Empty string field names are rejected."""
    with pytest.raises(TypeError, match="non-empty string keys"):

        @dataclass(frozen=True, slots=True)
        class _BadName(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"": 1}


def test_subclass_rejects_negative_axis_count() -> None:
    """Negative axis counts are rejected."""
    with pytest.raises(TypeError, match="must be an int >= 0"):

        @dataclass(frozen=True, slots=True)
        class _Negative(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": -1}


def test_subclass_rejects_unknown_field_reference() -> None:
    """``protected_axes`` referring to an unknown attribute is rejected."""
    with pytest.raises(TypeError, match="unknown field"):

        @dataclass(frozen=True, slots=True)
        class _Unknown(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"missing": 1}


def test_subclass_rejects_non_dict_permitted_ufuncs() -> None:
    """``permitted_ufuncs`` must be a dict."""
    with pytest.raises(TypeError, match="permitted_ufuncs must be a dict"):

        @dataclass(frozen=True, slots=True)
        class _BadUfuncs(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_ufuncs: ClassVar[Any] = [np.add]


def test_subclass_rejects_non_ufunc_key() -> None:
    """Keys of ``permitted_ufuncs`` must be ``np.ufunc`` instances."""
    with pytest.raises(TypeError, match="permitted_ufuncs keys must be numpy ufuncs"):

        @dataclass(frozen=True, slots=True)
        class _NonUfuncKey(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_ufuncs: ClassVar[Any] = {"not-a-ufunc": ["__call__"]}


def test_subclass_rejects_non_list_methods() -> None:
    """Values of ``permitted_ufuncs`` must be ``list[str]``."""
    with pytest.raises(TypeError, match="must be a list"):

        @dataclass(frozen=True, slots=True)
        class _BadMethods(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_ufuncs: ClassVar[Any] = {np.add: "__call__"}


def test_subclass_rejects_non_string_methods() -> None:
    """Methods inside ``permitted_ufuncs`` value lists must be strings."""
    with pytest.raises(TypeError, match="must be a list"):

        @dataclass(frozen=True, slots=True)
        class _NumberMethod(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_ufuncs: ClassVar[Any] = {np.add: [1]}


def test_subclass_rejects_non_set_permitted_functions() -> None:
    """``permitted_functions`` must be a set of callables."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _BadFuncs(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_functions: ClassVar[Any] = [np.mean]


def test_subclass_rejects_non_callable_in_permitted_functions() -> None:
    """Members of ``permitted_functions`` must be callable."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _NonCallable(ArrayAxisProtected[np.ndarray]):
            array: np.ndarray
            protected_axes: ClassVar[dict[str, int]] = {"array": 1}
            permitted_functions: ClassVar[set[Any]] = {"not-callable"}
