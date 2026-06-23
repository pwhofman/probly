"""Tests for the TorchSample Representation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.array_like import to_numpy_array_like
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.sample.array import ArraySample
from probly.representation.sample.torch import TorchSample
from probly.representation.torch_functions import torch_average


def _torch():
    """Return torch module or skip."""
    return pytest.importorskip("torch")


def assert_weights_equal(sample: TorchSample, expected: torch.Tensor) -> None:
    assert sample.weights is not None
    assert torch.equal(sample.weights, expected)


class TestTorchSample:
    def test_sample_internal_array(self, torch_tensor_sample_2d: TorchSample) -> None:
        assert isinstance(torch_tensor_sample_2d.tensor, torch.Tensor)

    def test_sample_length(self, torch_tensor_sample_2d: TorchSample) -> None:
        assert len(torch_tensor_sample_2d) == len(torch_tensor_sample_2d.tensor)

    def test_sample_iteration_follows_axis_zero(self) -> None:
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1)

        items = list(sample)

        assert len(items) == 3
        assert all(isinstance(item, TorchSample) for item in items)
        assert [item.sample_dim for item in items] == [0, 0, 0]
        assert torch.equal(items[0].tensor, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(items[1].tensor, torch.tensor([4, 5, 6, 7]))
        assert torch.equal(items[2].tensor, torch.tensor([8, 9, 10, 11]))

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

    def test_weighted_non_sample_slice_preserves_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        indexed_sample = sample[:2, :]

        assert isinstance(indexed_sample, TorchSample)
        assert indexed_sample.sample_dim == 1
        assert_weights_equal(indexed_sample, weights)

    def test_weighted_sample_dim_slice_indexes_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        indexed_sample = sample[:, 1:3]

        assert isinstance(indexed_sample, TorchSample)
        assert indexed_sample.sample_dim == 1
        assert_weights_equal(indexed_sample, torch.tensor([0.2, 0.3]))

    def test_weighted_integer_before_sample_dim_preserves_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        indexed_sample = sample[0, :]

        assert isinstance(indexed_sample, TorchSample)
        assert indexed_sample.sample_dim == 0
        assert_weights_equal(indexed_sample, weights)

    def test_weighted_integer_on_sample_dim_returns_tensor(self) -> None:
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.1, 0.2, 0.3, 0.4]))

        indexed_sample = sample[:, 2]

        assert isinstance(indexed_sample, torch.Tensor)

    def test_weighted_1d_integer_index_on_sample_dim_indexes_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        indexed_sample = sample[:, torch.tensor([3, 1])]

        assert isinstance(indexed_sample, TorchSample)
        assert_weights_equal(indexed_sample, torch.tensor([0.4, 0.2]))

    def test_weighted_1d_boolean_index_on_sample_dim_indexes_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        indexed_sample = sample[:, torch.tensor([True, False, True, False])]

        assert isinstance(indexed_sample, TorchSample)
        assert_weights_equal(indexed_sample, torch.tensor([0.1, 0.3]))

    def test_weighted_newaxis_preserves_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        indexed_sample = sample[None, :, :]

        assert isinstance(indexed_sample, TorchSample)
        assert indexed_sample.sample_dim == 2
        assert_weights_equal(indexed_sample, weights)

    def test_weighted_ellipsis_sample_dim_slice_indexes_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=weights)

        indexed_sample = sample[..., 1:3]

        assert isinstance(indexed_sample, TorchSample)
        assert_weights_equal(indexed_sample, torch.tensor([0.2, 0.3]))

    def test_weighted_multidimensional_integer_index_on_sample_dim_returns_tensor(self) -> None:
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1, weights=torch.tensor([0.1, 0.2, 0.3, 0.4]))

        indexed_sample = sample[:, torch.tensor([[0, 1]])]

        assert isinstance(indexed_sample, torch.Tensor)
        assert indexed_sample.shape == (3, 1, 2)

    def test_unweighted_complex_indexing_still_works(self) -> None:
        sample = TorchSample(torch.arange(12).reshape((3, 4)), sample_dim=1)

        indexed_sample = sample[:, torch.tensor([[0, 1]])]

        assert isinstance(indexed_sample, torch.Tensor)

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

    def test_sample_mean_uses_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12, dtype=torch.float32).reshape((3, 4)), sample_dim=1, weights=weights)

        result = sample.sample_mean()

        assert torch.allclose(result, torch_average(sample.tensor, dim=1, weights=weights))

    def test_sample_mean_of_categorical_distribution_preserves_distribution_type(self) -> None:
        probabilities = torch.arange(24, dtype=torch.float64).reshape((2, 3, 4)) + 1.0
        sample = TorchSample(TorchProbabilityCategoricalDistribution(probabilities), sample_dim=0)

        result = sample.sample_mean()

        assert isinstance(result, TorchCategoricalDistribution)
        assert result.shape == (3,)
        assert torch.allclose(result.unnormalized_probabilities, result.probabilities)
        assert torch.allclose(result.unnormalized_probabilities, torch.mean(sample.tensor.probabilities, dim=0))

    def test_weighted_sample_mean_of_categorical_distribution_uses_weights(self) -> None:
        probabilities = torch.arange(24, dtype=torch.float64).reshape((2, 3, 4)) + 1.0
        weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
        sample = TorchSample(TorchProbabilityCategoricalDistribution(probabilities), sample_dim=0, weights=weights)

        result = sample.sample_mean()

        assert isinstance(result, TorchCategoricalDistribution)
        assert result.shape == (3,)
        assert torch.allclose(
            result.unnormalized_probabilities,
            torch_average(sample.tensor.probabilities, dim=0, weights=weights),
        )

    def test_torch_average_supports_tuple_dim_weights(self) -> None:
        tensor = torch.arange(24, dtype=torch.float32).reshape((2, 3, 4))
        weights = torch.arange(1, 9, dtype=torch.float32).reshape((2, 4))
        expected = torch.sum(tensor * weights[:, None, :], dim=(0, 2)) / torch.sum(weights)

        result = torch_average(tensor, dim=(0, 2), weights=weights)

        assert torch.allclose(result, expected)

    def test_torch_average_supports_axis_alias(self) -> None:
        tensor = torch.arange(6, dtype=torch.float32).reshape((2, 3))

        result = torch_average(tensor, axis=1)

        assert torch.allclose(result, torch.mean(tensor, dim=1))

    def test_torch_average_dispatches_to_torch_sample(self) -> None:
        sample = TorchSample(torch.arange(12, dtype=torch.float32).reshape((3, 4)), sample_dim=1)

        result = torch_average(sample, dim=0)

        assert isinstance(result, TorchSample)
        assert result.sample_dim == 0
        assert torch.allclose(result.tensor, torch.mean(sample.tensor, dim=0))

    def test_sample_var_and_std_use_weights(self) -> None:
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample = TorchSample(torch.arange(12, dtype=torch.float32).reshape((3, 4)), sample_dim=1, weights=weights)
        average = torch_average(sample.tensor, dim=1, weights=weights, keepdim=True)
        expected_var = torch_average((sample.tensor - average) ** 2, dim=1, weights=weights)

        assert torch.allclose(sample.sample_var(), expected_var)
        assert torch.allclose(sample.sample_std(), torch.sqrt(expected_var))

    def test_weighted_sample_var_rejects_ddof(self) -> None:
        sample = TorchSample(torch.arange(12, dtype=torch.float32).reshape((3, 4)), sample_dim=1, weights=torch.ones(4))

        with pytest.raises(ValueError, match="ddof"):
            sample.sample_var(ddof=1)

    def test_to_numpy_array_like_uses_array_like(self, torch_tensor_sample_2d: TorchSample) -> None:
        converted = to_numpy_array_like(torch_tensor_sample_2d)

        assert isinstance(converted, ArraySample)
        assert converted.sample_axis == torch_tensor_sample_2d.sample_dim

    def test_torch_function_is_not_implemented(self, torch_tensor_sample_2d: TorchSample) -> None:
        result = TorchSample.__torch_function__(
            torch.prod,
            (TorchSample,),
            (torch_tensor_sample_2d,),
            {},
        )

        assert result is NotImplemented

    def test_torch_function_mean_reduces_sample_dim(self) -> None:
        sample = TorchSample(torch.arange(12, dtype=torch.float32).reshape((3, 4)), sample_dim=1)

        result = torch.mean(sample, dim=1)

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.mean(sample.tensor, dim=1))

    def test_torch_function_sum_tracks_sample_dim(self, torch_tensor_sample_2d: TorchSample) -> None:
        result = torch.sum(torch_tensor_sample_2d, dim=0)

        assert isinstance(result, TorchSample)
        assert result.sample_dim == 0
        assert torch.equal(result.tensor, torch.sum(torch_tensor_sample_2d.tensor, dim=0))

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


class TestTorchSampleEdgeCases:
    """TorchSample validation, mT/mH, conversions."""

    def test_invalid_sample_dim(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.zeros((2, 3))
        with pytest.raises(ValueError, match="out of bounds"):
            TorchSample(t, sample_dim=2)

    def test_negative_sample_dim_normalised(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.zeros((2, 3))
        s = TorchSample(t, sample_dim=-1)
        assert s.sample_dim == 1

    def test_negative_sample_dim_too_negative(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.zeros((2, 3))
        with pytest.raises(ValueError, match="out of bounds"):
            TorchSample(t, sample_dim=-3)

    def test_weights_shape_mismatch(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        with pytest.raises(ValueError, match="weights must have shape"):
            TorchSample(torch.zeros((2, 3)), sample_dim=0, weights=torch.zeros(5))

    def test_mT_swaps_sample_dim_when_at_end(self) -> None:  # noqa: N802
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.zeros((2, 3, 4))
        s = TorchSample(t, sample_dim=2)
        result = s.mT
        assert result.sample_dim == 1

    def test_mT_unaffected_when_other_dim(self) -> None:  # noqa: N802
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.zeros((2, 3, 4))
        s = TorchSample(t, sample_dim=0)
        result = s.mT
        assert result.sample_dim == 0

    def test_mT_requires_at_least_two_dims(self) -> None:  # noqa: N802
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(torch.zeros(5), sample_dim=0)
        with pytest.raises(ValueError, match="mT requires"):
            _ = s.mT

    def test_mH_swaps_sample_dim_when_at_end(self) -> None:  # noqa: N802
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.zeros((2, 3, 4))
        s = TorchSample(t, sample_dim=2)
        result = s.mH
        assert result.sample_dim == 1

    def test_mH_requires_at_least_two_dims(self) -> None:  # noqa: N802
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(torch.zeros(5), sample_dim=0)
        with pytest.raises(ValueError, match="mH requires"):
            _ = s.mH

    def test_size_with_dim(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.zeros((2, 3, 4))
        s = TorchSample(t, sample_dim=0)
        assert s.size(1) == 3
        assert tuple(s.size()) == (2, 3, 4)

    def test_samples_property_moves_axis(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.arange(24).reshape(2, 3, 4)
        s = TorchSample(t, sample_dim=2)
        out = s.samples
        # samples puts the sample axis first
        assert tuple(out.shape) == (4, 2, 3)

    def test_to_with_no_change_returns_same_object(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.zeros((2, 3))
        s = TorchSample(t, sample_dim=0)
        # to() with the same dtype/device shouldn't materialise a new wrapper.
        s2 = s.to(dtype=t.dtype, device=t.device)
        assert s2 is s

    def test_numpy_conversion(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(torch.arange(6, dtype=torch.float32).reshape(2, 3), sample_dim=0)
        arr = s.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3)

    def test_detach_returns_new_torch_sample(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        t = torch.zeros((3,), requires_grad=True)
        s = TorchSample(t, sample_dim=0)
        d = s.detach()
        assert d is not s
        assert not d.tensor.requires_grad

    def test_from_iterable_auto_axis_for_array_input(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        # Pass in a torch tensor directly with auto sample_axis.
        t = torch.arange(12).reshape(3, 4)
        s = TorchSample.from_iterable(t)
        # auto -> -1
        assert s.sample_dim == t.ndim - 1

    def test_from_iterable_auto_axis_zero_dim_raises(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        with pytest.raises(ValueError, match="Cannot infer"):
            TorchSample.from_iterable(torch.tensor(5))

    def test_from_iterable_auto_axis_empty_iterable_raises(self) -> None:
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        with pytest.raises(ValueError, match="Cannot infer"):
            TorchSample.from_iterable([])

    def test_from_iterable_both_dim_and_axis_raises(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        with pytest.raises(ValueError, match="Cannot specify both"):
            TorchSample.from_iterable(torch.zeros((2, 3)), sample_dim=0, sample_axis=1)

    def test_from_iterable_neither_dim_nor_axis_raises(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        with pytest.raises(ValueError, match="Either sample_dim or sample_axis"):
            TorchSample.from_iterable(torch.zeros((2, 3)), sample_dim=None, sample_axis=None)

    def test_to_array_like(self) -> None:
        torch = _torch()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(torch.arange(6, dtype=torch.float32).reshape(2, 3), sample_dim=0)
        arr_like = s.__array_like__()
        # Should produce an ArraySample
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        assert isinstance(arr_like, ArraySample)


def _torch_modules():
    """Return torch module or skip."""
    pytest.importorskip("torch")
    import torch as _torch  # noqa: PLC0415

    return _torch


class TestTorchSampleConcat:
    """TorchSample.concat with both Torch and non-Torch samples."""

    def test_concat_two_torch_samples(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        a = TorchSample(tensor=torch.tensor([[1.0, 2.0], [3.0, 4.0]]), sample_dim=0)
        b = TorchSample(tensor=torch.tensor([[5.0, 6.0], [7.0, 8.0]]), sample_dim=0)
        result = a.concat(b)
        assert result.tensor.shape == (4, 2)

    def test_concat_with_weights(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        a = TorchSample(
            tensor=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            sample_dim=0,
            weights=torch.tensor([0.5, 0.5]),
        )
        b = TorchSample(tensor=torch.tensor([[5.0, 6.0], [7.0, 8.0]]), sample_dim=0)
        result = a.concat(b)
        # b had no weights -> filled with 1.0.
        torch.testing.assert_close(result.weights, torch.tensor([0.5, 0.5, 1.0, 1.0]))

    def test_concat_with_other_weights(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        a = TorchSample(tensor=torch.tensor([[1.0, 2.0]]), sample_dim=0)
        b = TorchSample(
            tensor=torch.tensor([[3.0, 4.0]]),
            sample_dim=0,
            weights=torch.tensor([0.7]),
        )
        result = a.concat(b)
        torch.testing.assert_close(result.weights, torch.tensor([1.0, 0.7]))

    def test_move_sample_dim(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(tensor=torch.arange(24).reshape(2, 3, 4), sample_dim=0)
        moved = s.move_sample_dim(2)
        assert moved.sample_dim == 2
        assert tuple(moved.tensor.shape) == (3, 4, 2)

    def test_move_sample_axis_alias(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(tensor=torch.arange(24).reshape(2, 3, 4), sample_dim=0)
        moved = s.move_sample_axis(2)
        assert moved.sample_dim == 2

    def test_setitem(self) -> None:
        torch = _torch_modules()
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        s = TorchSample(tensor=torch.zeros(3, 2), sample_dim=0)
        s[0] = torch.tensor([1.0, 2.0])
        torch.testing.assert_close(s.tensor[0], torch.tensor([1.0, 2.0]))
