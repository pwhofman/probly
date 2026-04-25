"""Torch-specific tests for planned multi-library axis tracking behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.representation.array_like import ToIndices
from probly.representation.sample.axis_tracking import track_axis as track_axis_result


def track_axis(index: ToIndices, special_axis: int, ndim: int, torch_indexing: bool = False) -> object:
    result = track_axis_result(index, special_axis, ndim, torch_indexing=torch_indexing)
    return None if result is None else result.new_axis


def weight_index(index: ToIndices, special_axis: int, ndim: int, torch_indexing: bool = True) -> object:
    result = track_axis_result(index, special_axis, ndim, torch_indexing=torch_indexing)
    assert result is not None
    return result.index


class TestTensorIndexSemantics:
    def test_torch_tensor_index_supported_in_numpy_jax_mode(self) -> None:
        idx = (0, slice(None), torch.tensor([0, 2]))

        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=False) == 1
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=False) == 0

    def test_torch_tensor_index_supported_in_torch_mode(self) -> None:
        idx = (0, slice(None), torch.tensor([0, 2]))

        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=True) == 0
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=True) == 1


class TestScalarIndexSemantics:
    def test_0d_integer_tensor_is_treated_like_python_int(self) -> None:
        idx = (torch.tensor(0), slice(None), torch.tensor([0, 2]))

        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=True) == 0
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=True) == 1

    def test_0d_bool_tensor_is_not_treated_like_integer(self) -> None:
        idx = torch.tensor(True)

        assert track_axis(idx, special_axis=0, ndim=2, torch_indexing=True) == 1
        assert track_axis(idx, special_axis=1, ndim=2, torch_indexing=True) == 2


class TestTorchModeSemantics:
    def test_int_and_2d_advanced_index_keeps_unindexed_axis(self) -> None:
        idx = (0, torch.tensor([[0, 2], [1, 0]], dtype=torch.long))

        assert track_axis(idx, special_axis=2, ndim=4, torch_indexing=True) == 2

    def test_int_and_scalar_bool_index_keeps_remaining_axes(self) -> None:
        idx = (0, True)

        assert track_axis(idx, special_axis=1, ndim=4, torch_indexing=True) == 1
        assert track_axis(idx, special_axis=2, ndim=4, torch_indexing=True) == 2
        assert track_axis(idx, special_axis=3, ndim=4, torch_indexing=True) == 3

    def test_trailing_scalar_bool_with_ellipsis_keeps_last_axis(self) -> None:
        idx = (Ellipsis, True)

        assert track_axis(idx, special_axis=0, ndim=4, torch_indexing=True) == 0
        assert track_axis(idx, special_axis=1, ndim=4, torch_indexing=True) == 1
        assert track_axis(idx, special_axis=2, ndim=4, torch_indexing=True) == 2
        assert track_axis(idx, special_axis=3, ndim=4, torch_indexing=True) == 3

    def test_empty_ellipsis_does_not_separate_advanced_indices(self) -> None:
        idx = (slice(None), torch.tensor([0, 1]), Ellipsis, torch.tensor([0, 1]))

        assert track_axis(idx, special_axis=0, ndim=3, torch_indexing=True) == 0
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=True) == 1
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=True) == 1


class TestEllipsisSemantics:
    def test_ellipsis_with_nd_boolean_tensor_index(self) -> None:
        idx = (torch.tensor([[True, False, True], [False, True, False]]), Ellipsis)

        assert track_axis(idx, special_axis=0, ndim=3, torch_indexing=True) == 0
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=True) == 0
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=True) == 1

    def test_ellipsis_with_leading_newaxis_and_trailing_1d_integer_tensor_index(self) -> None:
        idx = (None, Ellipsis, torch.tensor([0, 2]))

        assert track_axis(idx, special_axis=0, ndim=3, torch_indexing=True) == 1
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=True) == 2
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=True) == 3


class TestTensorSubclassSupport:
    def test_fake_tensor_integer_array_index_supported(self) -> None:
        try:
            from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"FakeTensorMode unavailable: {exc}")

        with FakeTensorMode():
            x = torch.zeros((2, 3, 4))
            idx = torch.tensor([0, 2], dtype=torch.long)

            # Ensure this index type is accepted by torch itself.
            _ = x[(0, slice(None), idx)]

            assert track_axis((0, slice(None), idx), special_axis=1, ndim=3, torch_indexing=True) == 0
            assert track_axis((0, slice(None), idx), special_axis=2, ndim=3, torch_indexing=True) == 1

    def test_fake_tensor_0d_bool_index_supported(self) -> None:
        try:
            from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"FakeTensorMode unavailable: {exc}")

        with FakeTensorMode():
            x = torch.zeros((2, 3))
            idx = torch.tensor(True)

            # Ensure this index type is accepted by torch itself.
            _ = x[idx]

            assert track_axis(idx, special_axis=0, ndim=2, torch_indexing=True) == 1
            assert track_axis(idx, special_axis=1, ndim=2, torch_indexing=True) == 2


class TestWeightIndexTracking:
    def test_1d_integer_tensor_on_sample_axis_indexes_weights(self) -> None:
        idx = torch.tensor([3, 1])

        result = weight_index((slice(None), idx), special_axis=1, ndim=2)

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, idx)

    def test_1d_boolean_tensor_on_sample_axis_indexes_weights(self) -> None:
        idx = torch.tensor([True, False, True, False])

        result = weight_index((slice(None), idx), special_axis=1, ndim=2)

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, idx)
