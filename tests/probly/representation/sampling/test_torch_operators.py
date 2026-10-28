"""Torch sample operators preserve sample metadata and native tensor semantics."""

from __future__ import annotations

import operator

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution
from probly.representation.sample.torch import TorchSample


@pytest.mark.parametrize(
    "operation",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
        operator.and_,
        operator.or_,
        operator.xor,
        operator.lshift,
        operator.rshift,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("other_kind", ["scalar", "tensor", "sample", "numpy", "numpy_scalar"])
def test_binary_operators(operation, reflected, other_kind):
    tensor = torch.arange(1, 13).reshape(3, 4)
    weights = torch.arange(1, 5, dtype=torch.float32)
    sample = TorchSample(tensor, sample_dim=1, weights=weights)
    native_other = 2 if other_kind == "scalar" else torch.arange(1, 5)
    if other_kind == "numpy":
        other = np.arange(1, 5)
        result = operation(other, sample) if reflected else operation(sample, other)
        expected = operation(other, tensor.numpy()) if reflected else operation(tensor.numpy(), other)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)
        return
    if other_kind == "numpy_scalar":
        native_other = np.int64(2)
    other = TorchSample(native_other, sample_dim=0) if other_kind == "sample" else native_other

    result = operation(other, sample) if reflected else operation(sample, other)
    expected = operation(native_other, tensor) if reflected else operation(tensor, native_other)

    assert isinstance(result, TorchSample)
    assert result.sample_dim == 1
    torch.testing.assert_close(result.tensor, expected)
    torch.testing.assert_close(result.weights, weights)


@pytest.mark.parametrize("operation", [operator.pos, operator.neg, operator.abs, operator.invert])
def test_unary_operators(operation):
    sample = TorchSample(torch.arange(12).reshape(3, 4), sample_dim=1, weights=torch.ones(4))

    result = operation(sample)

    assert isinstance(result, TorchSample)
    assert result.sample_dim == sample.sample_dim
    torch.testing.assert_close(result.tensor, operation(sample.tensor))
    torch.testing.assert_close(result.weights, sample.weights)


def test_broadcasting_shifts_sample_axis():
    sample = TorchSample(torch.ones(3, 4), sample_dim=1, weights=torch.arange(1.0, 5.0))

    result = torch.zeros(2, 3, 4) + sample

    assert isinstance(result, TorchSample)
    assert result.sample_dim == 2
    torch.testing.assert_close(result.tensor, torch.ones(2, 3, 4))
    torch.testing.assert_close(result.weights, sample.weights)


def test_broadcasting_single_sample_expands_weights():
    sample = TorchSample(torch.ones(3, 1), sample_dim=1, weights=torch.tensor([0.5]))

    result = sample + torch.zeros(3, 4)

    assert isinstance(result, TorchSample)
    torch.testing.assert_close(result.weights, torch.full((4,), 0.5))


def test_mismatched_sample_axes_return_native_tensor():
    tensor = torch.arange(12).reshape(3, 4)

    result = TorchSample(tensor, sample_dim=0) + TorchSample(tensor, sample_dim=1)

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, tensor * 2)


@pytest.mark.parametrize("matching", [False, True])
def test_sample_weight_compatibility(matching):
    tensor = torch.ones(3, 4)
    left = TorchSample(tensor, sample_dim=1, weights=torch.arange(1.0, 5.0))
    right = TorchSample(tensor, sample_dim=1, weights=left.weights.clone() if matching else torch.ones(4))

    if not matching:
        with pytest.raises(ValueError, match="matching sample weights"):
            operator.add(left, right)
    else:
        result = left + right
        torch.testing.assert_close(result.weights, left.weights)


def test_keyword_operands_and_out():
    sample = TorchSample(torch.ones(3, 4), sample_dim=1, weights=torch.ones(4))
    out = TorchSample(torch.empty(3, 4), sample_dim=1, weights=sample.weights)

    result = torch.add(input=sample, other=2, alpha=3, out=out)

    assert result is out
    torch.testing.assert_close(out.tensor, torch.full((3, 4), 7.0))
    torch.testing.assert_close(out.weights, sample.weights)


@pytest.mark.parametrize("name", ["add", "sub", "mul", "multiply", "true_divide", "floor_divide"])
def test_tensor_methods_match_namespace_functions(name):
    sample = TorchSample(torch.arange(1.0, 13.0).reshape(3, 4), 1, torch.arange(1.0, 5.0))
    tensor = torch.full((3, 4), 2.0)

    result = getattr(tensor, name)(other=sample)
    expected = getattr(torch, name)(tensor, sample.tensor)

    assert isinstance(result, TorchSample)
    assert result.sample_dim == 1
    torch.testing.assert_close(result.tensor, expected)
    torch.testing.assert_close(result.weights, sample.weights)


def test_divmod_and_augmented_assignment():
    sample = TorchSample(torch.arange(1, 13).reshape(3, 4), sample_dim=1, weights=torch.ones(4))
    original = sample

    quotient, remainder = divmod(sample, 3)
    sample += 2

    assert sample is not original
    for result, expected in [
        (quotient, original.tensor // 3),
        (remainder, original.tensor % 3),
        (sample, original.tensor + 2),
    ]:
        assert isinstance(result, TorchSample)
        assert result.sample_dim == 1
        torch.testing.assert_close(result.tensor, expected)
        torch.testing.assert_close(result.weights, original.weights)


def test_operators_preserve_autograd():
    tensor = torch.arange(1.0, 13.0).reshape(3, 4).requires_grad_()
    sample = TorchSample(tensor, sample_dim=1, weights=torch.ones(4))

    (sample * 2).sample_mean().sum().backward()

    torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 0.5))


@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("other_kind", ["scalar", "tensor"])
def test_categorical_arithmetic_remains_unsupported(reflected, other_kind):
    sample = TorchSample(TorchProbabilityCategoricalDistribution(torch.ones(3, 4, 2)), sample_dim=1)
    other = 2 if other_kind == "scalar" else torch.full((3, 4), 2.0)
    left, right = (other, sample) if reflected else (sample, other)

    with pytest.raises(TypeError):
        operator.mul(left, right)


@pytest.mark.parametrize("method", ["add", "sin", "reduce", "accumulate", "outer", "modf"])
def test_numpy_ufuncs_keep_native_numpy_results(method):
    array = np.arange(1.0, 13.0).reshape(3, 4)
    sample = TorchSample(torch.from_numpy(array), 1, torch.ones(4))

    def apply(value):
        if method == "add":
            return np.add(value, 2)
        if method == "reduce":
            return np.add.reduce(value, axis=1)
        if method == "accumulate":
            return np.add.accumulate(value, axis=0)
        if method == "outer":
            return np.multiply.outer(value, value)
        return getattr(np, method)(value)

    result, expected = apply(sample), apply(array)
    pairs = zip(result, expected, strict=True) if isinstance(result, tuple) else [(result, expected)]
    for actual, wanted in pairs:
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, wanted)


def test_numpy_ufunc_out_and_mixed_infix_operands():
    sample = TorchSample(torch.ones(3, 4), 1, torch.ones(4))
    array = np.full((3, 4), 2.0, dtype=np.float32)
    out = np.empty_like(array)

    assert np.add(sample, array, out=out) is out
    np.testing.assert_array_equal(out, array + 1)
    assert isinstance(array + sample, np.ndarray)
    np.testing.assert_array_equal(array + sample, array + 1)

    result = sample + array
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, array + 1)
