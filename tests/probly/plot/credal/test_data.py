"""Tests for probability-extraction helpers in ``probly.plot.credal._data``."""

from __future__ import annotations

import numpy as np
import pytest

from probly.plot.credal._data import (
    _flatten_batch,
    _get_unnormalized_probabilities,
    _to_numpy,
)
from probly.representation.credal_set.array import ArraySingletonCredalSet


class _FakeTensor:
    """Minimal tensor-like object exercising the ``_to_numpy`` detach branch."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class TestDataHelpers:
    """Backend-agnostic helpers used to back credal set plots."""

    def test_to_numpy_passes_arrays_through(self) -> None:
        arr = np.array([0.3, 0.7])
        np.testing.assert_array_equal(_to_numpy(arr), arr)

    def test_to_numpy_converts_sequences(self) -> None:
        result = _to_numpy([1.0, 2.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_to_numpy_detaches_tensor_like(self) -> None:
        # Objects exposing ``.detach`` go through the detach/cpu/numpy path.
        arr = np.array([[0.4, 0.6]])
        np.testing.assert_array_equal(_to_numpy(_FakeTensor(arr)), arr)

    def test_get_unnormalized_probabilities_dispatches_array_set(self) -> None:
        data = ArraySingletonCredalSet(array=np.array([[0.3, 0.7]]))
        np.testing.assert_allclose(_get_unnormalized_probabilities(data), [[0.3, 0.7]])

    def test_get_unnormalized_probabilities_unregistered_type_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="Cannot extract probabilities from"):
            _get_unnormalized_probabilities(object())

    def test_flatten_batch_without_reshape_raises(self) -> None:
        with pytest.raises(TypeError, match="is not a supported batched credal set"):
            _flatten_batch(object())
