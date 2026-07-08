"""Tests for the abstract ``NumpyArrayLikeImplementation`` fallbacks."""

from __future__ import annotations

import numpy as np
import pytest


class TestNumpyArrayLikeImplementationFallbacks:
    """Default error paths in NumpyArrayLikeImplementation (the abstract base)."""

    def _make_minimal_subclass(self):
        """Build a minimal concrete subclass that does not override the error paths."""
        from probly.representation.array_like import NumpyArrayLikeImplementation  # noqa: PLC0415

        class _Minimal(NumpyArrayLikeImplementation[np.ndarray]):
            @property
            def dtype(self):
                return np.float64

            @property
            def device(self):
                return "cpu"

            @property
            def ndim(self):
                return 1

            @property
            def shape(self):
                return (1,)

            @property
            def size(self):
                return 1

            @property
            def flags(self):
                return None

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # noqa: ANN002, ANN003, ANN204
                return NotImplemented

            def __array_function__(self, func, types, args, kwargs):  # noqa: ANN204
                return NotImplemented

            def to_device(self, device, /, *, stream=None):  # noqa: ARG002
                return self

        return _Minimal()

    def test_getitem_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="indexing"):
            _ = m[0]

    def test_setitem_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="item assignment"):
            m[0] = 1

    def test_index_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to index"):
            m.__index__()

    def test_int_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to int"):
            m.__int__()

    def test_bool_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to bool"):
            m.__bool__()

    def test_float_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to float"):
            m.__float__()

    def test_complex_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="conversion to complex"):
            m.__complex__()

    def test_len_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="len"):
            len(m)

    def test_iter_raises_not_implemented(self) -> None:
        m = self._make_minimal_subclass()
        with pytest.raises(NotImplementedError, match="iteration"):
            iter(m)
