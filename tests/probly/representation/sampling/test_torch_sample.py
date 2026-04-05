"""Tests for the TorchTensorSample Representation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.array_like import to_numpy_array_like
from probly.representation.sample.array import ArraySample
from probly.representation.sample.torch import TorchTensorSample


class TestTorchTensorSample:
    def test_sample_internal_array(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        assert isinstance(torch_tensor_sample_2d.tensor, torch.Tensor)

    def test_sample_length(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        assert len(torch_tensor_sample_2d) == len(torch_tensor_sample_2d.tensor)

    def test_sample_ndim(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        assert torch_tensor_sample_2d.ndim == 2

    def test_sample_shape(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        assert torch_tensor_sample_2d.shape == torch_tensor_sample_2d.tensor.shape

    def test_sample_move_dim(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        moved_sample = torch_tensor_sample_2d.move_sample_dim(0)
        assert isinstance(moved_sample, TorchTensorSample)
        assert moved_sample.sample_axis == 0
        assert (
            torch_tensor_sample_2d.shape[torch_tensor_sample_2d.sample_axis]
            == moved_sample.shape[moved_sample.sample_axis]
        )

    def test_sample_concat(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        res = torch_tensor_sample_2d.concat(torch_tensor_sample_2d.move_sample_dim(0))
        assert isinstance(res, TorchTensorSample)
        assert res.sample_axis == torch_tensor_sample_2d.sample_axis
        assert res.sample_size == 2 * torch_tensor_sample_2d.sample_size

    def test_sample_slicing(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        indexed_sample = torch_tensor_sample_2d[:, :3]

        assert isinstance(indexed_sample, TorchTensorSample)
        assert indexed_sample.sample_dim == 1
        assert indexed_sample.shape == (3, 3)

    def test_sample_selection(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        indexed_sample = torch_tensor_sample_2d[:, 3]

        assert isinstance(indexed_sample, torch.Tensor)
        assert indexed_sample.shape == (3,)

    def test_sample_integer_index_before_sample_dim_shifts_sample_dim(
        self, torch_tensor_sample_2d: TorchTensorSample
    ) -> None:
        indexed_sample = torch_tensor_sample_2d[0, :]

        assert isinstance(indexed_sample, TorchTensorSample)
        assert indexed_sample.sample_dim == 0
        assert indexed_sample.shape == (4,)

    def test_sample_mixed_indexing_uses_torch_axis_tracking(self) -> None:
        sample = TorchTensorSample(torch.arange(24).reshape((2, 3, 4)), sample_dim=2)
        index = (0, slice(None), torch.tensor([0, 2]))

        indexed_sample = sample[index]

        assert isinstance(indexed_sample, TorchTensorSample)
        assert indexed_sample.sample_dim == 1
        assert torch.equal(indexed_sample.tensor, sample.tensor[index])

    def test_sample_setitem(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        torch_tensor_sample_2d[:, 0] = -1

        assert torch.equal(
            torch_tensor_sample_2d.tensor[:, 0], torch.full((3,), -1, dtype=torch_tensor_sample_2d.dtype)
        )

    def test_array_like_conversion(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        converted = torch_tensor_sample_2d.__array_like__()

        assert isinstance(converted, ArraySample)
        assert converted.sample_axis == torch_tensor_sample_2d.sample_dim
        assert np.array_equal(np.asarray(converted.array), np.asarray(torch_tensor_sample_2d.tensor))

    def test_to_numpy_array_like_uses_array_like(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        converted = to_numpy_array_like(torch_tensor_sample_2d)

        assert isinstance(converted, ArraySample)
        assert converted.sample_axis == torch_tensor_sample_2d.sample_dim

    def test_torch_function_is_not_implemented(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        result = TorchTensorSample.__torch_function__(
            torch.mean,
            (TorchTensorSample,),
            (torch_tensor_sample_2d,),
            {},
        )

        assert result is NotImplemented

    def test_torch_function_transpose(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        result = torch.transpose(torch_tensor_sample_2d, 0, 1)

        assert isinstance(result, TorchTensorSample)
        assert result.sample_dim == 0
        assert torch.equal(result.tensor, torch.transpose(torch_tensor_sample_2d.tensor, 0, 1))

    def test_torch_function_permute(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        result = torch.permute(torch_tensor_sample_2d, (1, 0))

        assert isinstance(result, TorchTensorSample)
        assert result.sample_dim == 0
        assert torch.equal(result.tensor, torch.permute(torch_tensor_sample_2d.tensor, (1, 0)))

    def test_torch_function_adjoint(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        result = torch.adjoint(torch_tensor_sample_2d)

        assert isinstance(result, TorchTensorSample)
        assert result.sample_dim == 0
        assert torch.equal(result.tensor, torch.adjoint(torch_tensor_sample_2d.tensor))

    def test_torch_function_cat_preserves_type_for_matching_sample_dim(
        self, torch_tensor_sample_2d: TorchTensorSample
    ) -> None:
        result = torch.cat((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=1)

        assert isinstance(result, TorchTensorSample)
        assert result.sample_dim == 1
        assert torch.equal(
            result.tensor, torch.cat((torch_tensor_sample_2d.tensor, torch_tensor_sample_2d.tensor), dim=1)
        )

    def test_torch_function_cat_drops_type_for_mismatched_sample_dim(
        self, torch_tensor_sample_2d: TorchTensorSample
    ) -> None:
        other = TorchTensorSample(torch.arange(12, 24).reshape((3, 4)), sample_dim=0)
        result = torch.cat((torch_tensor_sample_2d, other), dim=0)

        assert isinstance(result, torch.Tensor)

    def test_torch_function_cat_returns_sample_out(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        out = TorchTensorSample(torch.empty((3, 8), dtype=torch_tensor_sample_2d.dtype), sample_dim=0)

        result = torch.cat((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=1, out=out)

        assert result is out
        assert torch.equal(out.tensor, torch.cat((torch_tensor_sample_2d.tensor, torch_tensor_sample_2d.tensor), dim=1))

    def test_torch_function_concat_aliases(self, torch_tensor_sample_2d: TorchTensorSample) -> None:
        result_concat = torch.concat((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=1)
        result_concatenate = torch.concatenate((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=1)

        assert isinstance(result_concat, TorchTensorSample)
        assert isinstance(result_concatenate, TorchTensorSample)
        assert result_concat.sample_dim == torch_tensor_sample_2d.sample_dim
        assert result_concatenate.sample_dim == torch_tensor_sample_2d.sample_dim
