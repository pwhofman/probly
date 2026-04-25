"""Tests for the TorchSample Representation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.array_like import to_numpy_array_like
from probly.representation.sample.array import ArraySample
from probly.representation.sample.torch import TorchSample


def assert_weights_equal(sample: TorchSample, expected: torch.Tensor) -> None:
    assert sample.weights is not None
    assert torch.equal(sample.weights, expected)


class TestTorchSample:
    def test_sample_internal_array(self, torch_tensor_sample_2d: TorchSample) -> None:
        assert isinstance(torch_tensor_sample_2d.tensor, torch.Tensor)

    def test_sample_length(self, torch_tensor_sample_2d: TorchSample) -> None:
        assert len(torch_tensor_sample_2d) == len(torch_tensor_sample_2d.tensor)

    def test_sample_ndim(self, torch_tensor_sample_2d: TorchSample) -> None:
        assert torch_tensor_sample_2d.ndim == 2

    def test_sample_shape(self, torch_tensor_sample_2d: TorchSample) -> None:
        assert torch_tensor_sample_2d.shape == torch_tensor_sample_2d.tensor.shape

    def test_sample_move_dim(self, torch_tensor_sample_2d: TorchSample) -> None:
        moved_sample = torch_tensor_sample_2d.move_sample_dim(0)
        assert isinstance(moved_sample, TorchSample)
        assert moved_sample.sample_axis == 0
        assert (
            torch_tensor_sample_2d.shape[torch_tensor_sample_2d.sample_axis]
            == moved_sample.shape[moved_sample.sample_axis]
        )

    def test_sample_concat(self, torch_tensor_sample_2d: TorchSample) -> None:
        res = torch_tensor_sample_2d.concat(torch_tensor_sample_2d.move_sample_dim(0))
        assert isinstance(res, TorchSample)
        assert res.sample_axis == torch_tensor_sample_2d.sample_axis
        assert res.sample_size == 2 * torch_tensor_sample_2d.sample_size

    def test_from_iterable_preserves_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3])

        sample = TorchSample.from_iterable(torch.arange(6).reshape((3, 2)), sample_axis=0, weights=weights)

        assert_weights_equal(sample, weights)

    def test_constructor_rejects_wrong_weight_shape(self) -> None:
        with pytest.raises(ValueError, match="weights must have shape"):
            TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.1, 0.2, 0.3]))

    def test_sample_move_dim_preserves_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        moved_sample = sample.move_sample_dim(0)

        assert moved_sample.sample_dim == 0
        assert_weights_equal(moved_sample, weights)

    def test_sample_concat_combines_weights(self) -> None:
        left = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        right = TorchSample(
            torch.arange(12, 24).reshape((4, 3)), sample_dim=0, weights=torch.tensor([0.5, 0.6, 0.7, 0.8])
        )

        result = left.concat(right)

        assert result.sample_dim == 1
        assert_weights_equal(result, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))

    def test_sample_concat_fills_missing_weights_with_ones(self) -> None:
        left = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1)
        right = TorchSample(
            torch.arange(12, 24).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.5, 0.6, 0.7, 0.8])
        )

        result = left.concat(right)

        assert_weights_equal(result, torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5, 0.6, 0.7, 0.8]))

    def test_sample_slicing(self, torch_tensor_sample_2d: TorchSample) -> None:
        indexed_sample = torch_tensor_sample_2d[:, :3]

        assert isinstance(indexed_sample, TorchSample)
        assert indexed_sample.sample_dim == 1
        assert indexed_sample.shape == (3, 3)

    def test_sample_selection(self, torch_tensor_sample_2d: TorchSample) -> None:
        indexed_sample = torch_tensor_sample_2d[:, 3]

        assert isinstance(indexed_sample, torch.Tensor)
        assert indexed_sample.shape == (3,)

    def test_sample_integer_index_before_sample_dim_shifts_sample_dim(
        self, torch_tensor_sample_2d: TorchSample
    ) -> None:
        indexed_sample = torch_tensor_sample_2d[0, :]

        assert isinstance(indexed_sample, TorchSample)
        assert indexed_sample.sample_dim == 0
        assert indexed_sample.shape == (4,)

    def test_sample_mixed_indexing_uses_torch_axis_tracking(self) -> None:
        sample = TorchSample(torch.arange(24).reshape((2, 3, 4)), sample_dim=2)
        index = (0, slice(None), torch.tensor([0, 2]))

        indexed_sample = sample[index]

        assert isinstance(indexed_sample, TorchSample)
        assert indexed_sample.sample_dim == 1
        assert torch.equal(indexed_sample.tensor, sample.tensor[index])

    def test_sample_setitem(self, torch_tensor_sample_2d: TorchSample) -> None:
        torch_tensor_sample_2d[:, 0] = -1

        assert torch.equal(
            torch_tensor_sample_2d.tensor[:, 0], torch.full((3,), -1, dtype=torch_tensor_sample_2d.dtype)
        )

    def test_array_like_conversion(self, torch_tensor_sample_2d: TorchSample) -> None:
        converted = torch_tensor_sample_2d.__array_like__()

        assert isinstance(converted, ArraySample)
        assert converted.sample_axis == torch_tensor_sample_2d.sample_dim
        assert np.array_equal(np.asarray(converted.array), np.asarray(torch_tensor_sample_2d.tensor))

    def test_detach_preserves_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        detached = sample.detach()

        assert detached.tensor is not sample.tensor
        assert_weights_equal(detached, weights)

    def test_to_preserves_weights(self) -> None:
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.1, 0.2, 0.3, 0.4]))

        converted = sample.to(dtype=torch.float64)

        assert converted.tensor.dtype == torch.float64
        assert converted.weights is not None
        assert converted.weights.dtype == torch.float64
        assert torch.allclose(converted.weights, torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64))

    def test_weighted_statistics_raise_not_implemented(self) -> None:
        sample = TorchSample(torch.arange(12, dtype=torch.float32).reshape((3, 4)), sample_dim=1, weights=torch.ones(4))

        with pytest.raises(NotImplementedError, match="Weighted sample mean"):
            sample.sample_mean()
        with pytest.raises(NotImplementedError, match="Weighted sample variance"):
            sample.sample_var()
        with pytest.raises(NotImplementedError, match="Weighted sample std"):
            sample.sample_std()

    def test_to_numpy_array_like_uses_array_like(self, torch_tensor_sample_2d: TorchSample) -> None:
        converted = to_numpy_array_like(torch_tensor_sample_2d)

        assert isinstance(converted, ArraySample)
        assert converted.sample_axis == torch_tensor_sample_2d.sample_dim

    def test_torch_function_is_not_implemented(self, torch_tensor_sample_2d: TorchSample) -> None:
        result = TorchSample.__torch_function__(
            torch.mean,
            (TorchSample,),
            (torch_tensor_sample_2d,),
            {},
        )

        assert result is NotImplemented

    def test_torch_function_transpose(self, torch_tensor_sample_2d: TorchSample) -> None:
        result = torch.transpose(torch_tensor_sample_2d, 0, 1)

        assert isinstance(result, TorchSample)
        assert result.sample_dim == 0
        assert torch.equal(result.tensor, torch.transpose(torch_tensor_sample_2d.tensor, 0, 1))

    def test_torch_function_transpose_preserves_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        result = torch.transpose(sample, 0, 1)

        assert isinstance(result, TorchSample)
        assert_weights_equal(result, weights)

    def test_torch_function_permute(self, torch_tensor_sample_2d: TorchSample) -> None:
        result = torch.permute(torch_tensor_sample_2d, (1, 0))

        assert isinstance(result, TorchSample)
        assert result.sample_dim == 0
        assert torch.equal(result.tensor, torch.permute(torch_tensor_sample_2d.tensor, (1, 0)))

    def test_torch_function_permute_preserves_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        result = torch.permute(sample, (1, 0))

        assert isinstance(result, TorchSample)
        assert_weights_equal(result, weights)

    def test_torch_function_adjoint(self, torch_tensor_sample_2d: TorchSample) -> None:
        result = torch.adjoint(torch_tensor_sample_2d)

        assert isinstance(result, TorchSample)
        assert result.sample_dim == 0
        assert torch.equal(result.tensor, torch.adjoint(torch_tensor_sample_2d.tensor))

    def test_torch_function_cat_preserves_type_for_matching_sample_dim(
        self, torch_tensor_sample_2d: TorchSample
    ) -> None:
        result = torch.cat((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=1)

        assert isinstance(result, TorchSample)
        assert result.sample_dim == 1
        assert torch.equal(
            result.tensor, torch.cat((torch_tensor_sample_2d.tensor, torch_tensor_sample_2d.tensor), dim=1)
        )

    def test_torch_function_cat_combines_weights(self) -> None:
        left = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        right = TorchSample(
            torch.arange(12, 24).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.5, 0.6, 0.7, 0.8])
        )

        result = torch.cat((left, right), dim=1)

        assert isinstance(result, TorchSample)
        assert_weights_equal(result, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))

    def test_torch_function_cat_fills_missing_weights_with_ones(self) -> None:
        left = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1)
        right = TorchSample(
            torch.arange(12, 24).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.5, 0.6, 0.7, 0.8])
        )

        result = torch.cat((left, right), dim=1)

        assert isinstance(result, TorchSample)
        assert_weights_equal(result, torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5, 0.6, 0.7, 0.8]))

    def test_torch_function_cat_weighted_non_sample_dim_raises(self) -> None:
        left = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        right = TorchSample(
            torch.arange(12, 24).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.5, 0.6, 0.7, 0.8])
        )

        with pytest.raises(ValueError, match="sample dimension"):
            torch.cat((left, right), dim=0)

    def test_torch_function_cat_drops_type_for_mismatched_sample_dim(self, torch_tensor_sample_2d: TorchSample) -> None:
        other = TorchSample(torch.arange(12, 24).reshape((3, 4)), sample_dim=0)
        result = torch.cat((torch_tensor_sample_2d, other), dim=0)

        assert isinstance(result, torch.Tensor)

    def test_torch_function_cat_returns_sample_out(self, torch_tensor_sample_2d: TorchSample) -> None:
        out = TorchSample(torch.empty((3, 8), dtype=torch_tensor_sample_2d.dtype), sample_dim=0)

        result = torch.cat((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=1, out=out)

        assert result is out
        assert torch.equal(out.tensor, torch.cat((torch_tensor_sample_2d.tensor, torch_tensor_sample_2d.tensor), dim=1))

    def test_torch_function_concat_aliases(self, torch_tensor_sample_2d: TorchSample) -> None:
        result_concat = torch.concat((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=1)
        result_concatenate = torch.concatenate((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=1)

        assert isinstance(result_concat, TorchSample)
        assert isinstance(result_concatenate, TorchSample)
        assert result_concat.sample_dim == torch_tensor_sample_2d.sample_dim
        assert result_concatenate.sample_dim == torch_tensor_sample_2d.sample_dim

    def test_torch_function_stack_shifts_sample_dim_when_dim_is_before(
        self, torch_tensor_sample_2d: TorchSample
    ) -> None:
        result = torch.stack((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=0)

        assert isinstance(result, TorchSample)
        assert result.sample_dim == torch_tensor_sample_2d.sample_dim + 1
        assert torch.equal(
            result.tensor, torch.stack((torch_tensor_sample_2d.tensor, torch_tensor_sample_2d.tensor), dim=0)
        )

    def test_torch_function_stack_keeps_sample_dim_when_dim_is_after(self, torch_tensor_sample_2d: TorchSample) -> None:
        result = torch.stack((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=2)

        assert isinstance(result, TorchSample)
        assert result.sample_dim == torch_tensor_sample_2d.sample_dim
        assert torch.equal(
            result.tensor, torch.stack((torch_tensor_sample_2d.tensor, torch_tensor_sample_2d.tensor), dim=2)
        )

    def test_torch_function_stack_drops_type_for_mismatched_sample_dim(
        self, torch_tensor_sample_2d: TorchSample
    ) -> None:
        other = TorchSample(torch.arange(12, 24).reshape((3, 4)), sample_dim=0)
        result = torch.stack((torch_tensor_sample_2d, other), dim=0)

        assert isinstance(result, torch.Tensor)

    def test_torch_function_stack_with_sample_out_returns_out(self, torch_tensor_sample_2d: TorchSample) -> None:
        out = TorchSample(torch.empty((2, 3, 4), dtype=torch_tensor_sample_2d.dtype), sample_dim=0)

        result = torch.stack((torch_tensor_sample_2d, torch_tensor_sample_2d), dim=0, out=out)

        assert result is out
        assert torch.equal(
            out.tensor, torch.stack((torch_tensor_sample_2d.tensor, torch_tensor_sample_2d.tensor), dim=0)
        )

    def test_torch_function_stack_with_weights_raises(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        left = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)
        right = TorchSample(torch.arange(12, 24).reshape((3, 4)), sample_dim=1, weights=weights)

        with pytest.raises(ValueError, match="stack"):
            torch.stack((left, right), dim=0)
