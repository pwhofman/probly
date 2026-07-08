"""Tests for ``TorchAxisProtected`` ``__init_subclass__`` validation paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

pytest.importorskip("torch")
import torch

from probly.representation._protected_axis.torch import (
    TorchAxisProtected,
    _validate_field_ndim,
)


def test_validate_field_ndim_raises_when_too_low() -> None:
    """Helper raises ValueError when ndim is less than required protected axes."""
    with pytest.raises(ValueError, match="ndim 1, expected >= 2"):
        _validate_field_ndim("field", 1, 2)


def test_torch_subclass_requires_non_empty_protected_axes() -> None:
    """Torch subclasses must define ``protected_axes`` as a non-empty dict."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _Empty(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {}


def test_torch_subclass_rejects_non_dict_protected_axes() -> None:
    """Torch subclasses with non-dict ``protected_axes`` raise TypeError."""
    with pytest.raises(TypeError, match="must define protected_axes"):

        @dataclass(frozen=True, slots=True)
        class _NotDict(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[Any] = ["tensor"]


def test_torch_subclass_rejects_empty_string_field_name() -> None:
    """Empty string field names are rejected for torch subclasses."""
    with pytest.raises(TypeError, match="non-empty string keys"):

        @dataclass(frozen=True, slots=True)
        class _BadName(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {"": 1}


def test_torch_subclass_rejects_negative_axis_count() -> None:
    """Negative axis counts are rejected."""
    with pytest.raises(TypeError, match="must be an int >= 0"):

        @dataclass(frozen=True, slots=True)
        class _Negative(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {"tensor": -1}


def test_torch_subclass_rejects_non_int_axis_count() -> None:
    """Non-int axis counts are rejected."""
    with pytest.raises(TypeError, match="must be an int >= 0"):

        @dataclass(frozen=True, slots=True)
        class _Fractional(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[Any] = {"tensor": 1.5}


def test_torch_subclass_rejects_unknown_field_reference() -> None:
    """``protected_axes`` referring to an unknown attribute is rejected."""
    with pytest.raises(TypeError, match="unknown field"):

        @dataclass(frozen=True, slots=True)
        class _Unknown(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {"missing": 1}


def test_torch_subclass_rejects_non_set_permitted_functions() -> None:
    """``permitted_functions`` must be a set of callables."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _BadFuncs(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}
            permitted_functions: ClassVar[Any] = [torch.mean]


def test_torch_subclass_rejects_non_callable_in_permitted_functions() -> None:
    """Members of ``permitted_functions`` must be callable."""
    with pytest.raises(TypeError, match="permitted_functions must be a set"):

        @dataclass(frozen=True, slots=True)
        class _NonCallable(TorchAxisProtected[Any]):
            tensor: torch.Tensor
            protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}
            permitted_functions: ClassVar[set[Any]] = {"not-callable"}
