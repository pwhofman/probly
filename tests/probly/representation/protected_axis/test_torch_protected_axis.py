"""Tests for generalized torch protected-axis behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation._protected_axis.torch import TorchAxisProtected


@dataclass(frozen=True, slots=True)
class SingleTensor(TorchAxisProtected[Any]):
    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}


@dataclass(frozen=True, slots=True)
class ReductionTensor(TorchAxisProtected[Any]):
    tensor: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}
    permitted_functions: ClassVar[set[Any]] = {torch.mean, torch.sum}


@dataclass(frozen=True, slots=True)
class PairTensor(TorchAxisProtected[Any]):
    first: torch.Tensor
    second: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"first": 0, "second": 0}


@dataclass(frozen=True, slots=True)
class InnerPairTensor(TorchAxisProtected[Any]):
    left: torch.Tensor
    right: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}


@dataclass(frozen=True, slots=True)
class OuterNestedTensor(TorchAxisProtected[Any]):
    inner: InnerPairTensor
    aux: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "aux": 1}


def test_single_field_torch_functions_preserve_type() -> None:
    x = SingleTensor(torch.arange(6.0).reshape(2, 3))

    expanded = torch.unsqueeze(x, dim=0)
    assert isinstance(expanded, SingleTensor)
    assert tuple(expanded.tensor.shape) == (1, 2, 3)

    stacked = torch.stack((x, x), dim=0)
    assert isinstance(stacked, SingleTensor)
    assert tuple(stacked.tensor.shape) == (2, 2, 3)


def test_reshape_method_preserves_protected_axes() -> None:
    x = SingleTensor(torch.arange(24.0).reshape(2, 3, 4))

    reshaped = x.reshape(6)
    assert isinstance(reshaped, SingleTensor)
    assert tuple(reshaped.tensor.shape) == (6, 4)
    assert reshaped.shape == (6,)
    assert reshaped.protected_shape == (4,)

    reshaped_with_tuple = x.reshape((1, 6))
    assert isinstance(reshaped_with_tuple, SingleTensor)
    assert tuple(reshaped_with_tuple.tensor.shape) == (1, 6, 4)
    assert reshaped_with_tuple.shape == (1, 6)
    assert reshaped_with_tuple.protected_shape == (4,)


def test_unpermitted_torch_reductions_return_notimplemented() -> None:
    x = SingleTensor(torch.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError):
        _ = torch.sum(x)

    with pytest.raises(TypeError):
        _ = torch.mean(x)


def test_permitted_torch_reductions_reduce_only_batch_axes() -> None:
    x = ReductionTensor(torch.arange(24.0).reshape(2, 3, 4))

    summed = torch.sum(x)
    assert isinstance(summed, ReductionTensor)
    assert torch.equal(summed.tensor, torch.sum(x.tensor, dim=(0, 1)))
    assert summed.shape == ()
    assert summed.protected_shape == (4,)

    meaned = torch.mean(x, dim=1)
    assert isinstance(meaned, ReductionTensor)
    assert torch.equal(meaned.tensor, torch.mean(x.tensor, dim=1))
    assert meaned.shape == (2,)
    assert meaned.protected_shape == (4,)


def test_multi_field_tensor_conversion_and_cat() -> None:
    x = PairTensor(torch.ones((2, 2)), torch.ones((2, 2)) * 2)

    with pytest.raises(TypeError, match="Cannot convert multi-field"):
        _ = np.asarray(x)

    cat = torch.cat((x, x), dim=0)
    assert isinstance(cat, PairTensor)
    assert tuple(cat.first.shape) == (4, 2)
    assert tuple(cat.second.shape) == (4, 2)


def test_reshape_method_applies_to_all_fields() -> None:
    x = PairTensor(torch.arange(6.0).reshape(2, 3), torch.arange(6.0).reshape(2, 3) + 10)

    reshaped = x.reshape(3, 2)
    assert isinstance(reshaped, PairTensor)
    assert tuple(reshaped.first.shape) == (3, 2)
    assert tuple(reshaped.second.shape) == (3, 2)


def test_multi_field_index_and_assignment() -> None:
    x = PairTensor(torch.zeros((2, 2)), torch.ones((2, 2)))

    y = x[0]
    assert isinstance(y, PairTensor)
    assert tuple(y.first.shape) == (2,)
    assert tuple(y.second.shape) == (2,)

    x[0] = PairTensor(torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0]))
    assert torch.equal(x.first[0], torch.tensor([3.0, 4.0]))
    assert torch.equal(x.second[0], torch.tensor([5.0, 6.0]))

    x[1] = (torch.tensor([7.0, 8.0]), torch.tensor([9.0, 10.0]))
    assert torch.equal(x.first[1], torch.tensor([7.0, 8.0]))
    assert torch.equal(x.second[1], torch.tensor([9.0, 10.0]))


def test_nested_multi_field_value_delegates_without_tensor_casts() -> None:
    inner = InnerPairTensor(
        left=torch.arange(24.0).reshape(2, 3, 4),
        right=torch.arange(24.0).reshape(2, 3, 4) + 100,
    )
    outer = OuterNestedTensor(inner=inner, aux=torch.arange(10.0).reshape(2, 5))

    values = outer.protected_values()
    assert values["inner"] is inner

    expanded = torch.unsqueeze(outer, dim=0)
    assert isinstance(expanded, OuterNestedTensor)
    assert isinstance(expanded.inner, InnerPairTensor)
    assert tuple(expanded.inner.left.shape) == (1, 2, 3, 4)
    assert tuple(expanded.inner.right.shape) == (1, 2, 3, 4)
    assert tuple(expanded.aux.shape) == (1, 2, 5)

    stacked = torch.stack((outer, outer), dim=0)
    assert isinstance(stacked, OuterNestedTensor)
    assert isinstance(stacked.inner, InnerPairTensor)
    assert tuple(stacked.inner.left.shape) == (2, 2, 3, 4)
    assert tuple(stacked.inner.right.shape) == (2, 2, 3, 4)
    assert tuple(stacked.aux.shape) == (2, 2, 5)

    sliced = outer[1]
    assert isinstance(sliced, OuterNestedTensor)
    assert isinstance(sliced.inner, InnerPairTensor)
    assert tuple(sliced.inner.left.shape) == (3, 4)
    assert tuple(sliced.inner.right.shape) == (3, 4)
    assert tuple(sliced.aux.shape) == (5,)
