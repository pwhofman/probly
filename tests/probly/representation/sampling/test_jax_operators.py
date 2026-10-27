"""JAX sample operators and explicitly declared dynamic fields."""

from __future__ import annotations

import operator

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
from jax.experimental import checkify
import jax.numpy as jnp
from ndlike.jax_functions import jax_add, jax_multiply, jax_subtract, jax_true_divide
from ndlike.jax_like import jax_tree_map

from probly.representation.distribution.jax_categorical import JaxProbabilityCategoricalDistribution
from probly.representation.sample.jax import JaxSample


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
@pytest.mark.parametrize("other_kind", ["scalar", "array", "sample", "numpy", "numpy_scalar"])
def test_binary_operators(operation, reflected, other_kind):
    array = jnp.arange(1, 13).reshape(3, 4)
    weights = jnp.arange(1.0, 5.0)
    sample = JaxSample(array, sample_axis=1, weights=weights)
    native_other = 2 if other_kind == "scalar" else jnp.arange(1, 5)
    if other_kind == "numpy":
        other = np.arange(1, 5, dtype=np.int32)
        result = operation(other, sample) if reflected else operation(sample, other)
        expected = operation(other, np.asarray(array)) if reflected else operation(np.asarray(array), other)
        if reflected:
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, expected)
        else:
            # JAX accepts NumPy arrays as inputs to its own operations.
            assert isinstance(result, JaxSample)
            np.testing.assert_allclose(result.array, expected)
            np.testing.assert_array_equal(result.weights, weights)
        return
    if other_kind == "numpy_scalar":
        native_other = np.int64(2)
    other = JaxSample(native_other, sample_axis=0) if other_kind == "sample" else native_other

    result = operation(other, sample) if reflected else operation(sample, other)
    expected = operation(native_other, array) if reflected else operation(array, native_other)

    # Native JAX left operands consume __jax_array__, just as before the upgrade.
    if reflected and other_kind == "array":
        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(result, expected)
        return

    assert isinstance(result, JaxSample)
    assert result.sample_axis == 1
    np.testing.assert_allclose(result.array, expected)
    np.testing.assert_array_equal(result.weights, weights)


@pytest.mark.parametrize("operation", [operator.pos, operator.neg, operator.abs, operator.invert])
def test_unary_operators(operation):
    sample = JaxSample(jnp.arange(12).reshape(3, 4), sample_axis=1, weights=jnp.ones(4))

    result = operation(sample)

    assert isinstance(result, JaxSample)
    assert result.sample_axis == sample.sample_axis
    np.testing.assert_array_equal(result.array, operation(sample.array))
    np.testing.assert_array_equal(result.weights, sample.weights)


def test_broadcasting_shifts_sample_axis():
    sample = JaxSample(jnp.ones((3, 4)), sample_axis=1, weights=jnp.arange(1.0, 5.0))

    result = sample + jnp.zeros((2, 3, 4))

    assert isinstance(result, JaxSample)
    assert result.sample_axis == 2
    np.testing.assert_array_equal(result.array, jnp.ones((2, 3, 4)))
    np.testing.assert_array_equal(result.weights, sample.weights)


def test_mismatched_sample_axes_return_native_array():
    array = jnp.arange(12).reshape(3, 4)

    result = JaxSample(array, sample_axis=0) + JaxSample(array, sample_axis=1)

    assert isinstance(result, jax.Array)
    np.testing.assert_array_equal(result, array * 2)


def test_explicit_wrapper_matches_infix_operation():
    sample = JaxSample(jnp.ones((3, 4)), sample_axis=1, weights=jnp.ones(4))

    result = jax_add(sample, 2)

    assert isinstance(result, JaxSample)
    np.testing.assert_array_equal(result.array, jnp.full((3, 4), 3.0))
    np.testing.assert_array_equal(result.weights, sample.weights)


@pytest.mark.parametrize("function", [jax_add, jax_subtract, jax_multiply, jax_true_divide])
@pytest.mark.parametrize("reflected", [False, True])
def test_jax_wrappers_preserve_sample_in_both_operand_orders(function, reflected):
    sample = JaxSample(jnp.arange(1.0, 13.0).reshape(3, 4), 1, jnp.arange(1.0, 5.0))
    array = jnp.full((2, 3, 4), 2.0)

    result = function(array, sample) if reflected else function(sample, array)
    expected = function(array, sample.array) if reflected else function(sample.array, array)

    assert isinstance(result, JaxSample)
    assert result.sample_axis == 2
    np.testing.assert_allclose(result.array, expected)
    np.testing.assert_array_equal(result.weights, sample.weights)


@pytest.mark.parametrize("reflected", [False, True])
def test_jax_wrapper_does_not_coerce_unsupported_nested_operation(reflected):
    sample = JaxSample(JaxProbabilityCategoricalDistribution(jnp.ones((3, 4, 2))), 1)
    other = jnp.full((3, 4), 2.0)
    left, right = (other, sample) if reflected else (sample, other)

    with pytest.raises(TypeError):
        jax_multiply(left, right)


def test_divmod_and_augmented_assignment():
    sample = JaxSample(jnp.arange(1, 13).reshape(3, 4), sample_axis=1, weights=jnp.ones(4))
    original = sample

    quotient, remainder = divmod(sample, 3)
    sample += 2

    assert sample is not original
    for result, expected in [
        (quotient, original.array // 3),
        (remainder, original.array % 3),
        (sample, original.array + 2),
    ]:
        assert isinstance(result, JaxSample)
        assert result.sample_axis == 1
        np.testing.assert_array_equal(result.array, expected)
        np.testing.assert_array_equal(result.weights, original.weights)


@pytest.mark.parametrize("weighted", [False, True])
def test_jit_preserves_axis_and_dynamic_weights(weighted):
    sample = JaxSample(jnp.arange(12.0).reshape(3, 4), 1, jnp.arange(1.0, 5.0) if weighted else None)
    double = jax.jit(lambda value: value * 2)

    for offset in (0, 2):
        current = JaxSample(sample.array, 1, sample.weights + offset if weighted else None)
        result = double(current)
        assert isinstance(result, JaxSample)
        assert result.sample_axis == 1
        np.testing.assert_array_equal(result.array, current.array * 2)
        np.testing.assert_array_equal(result.weights, current.weights)
        np.testing.assert_allclose(result.sample_mean(), current.sample_mean() * 2)


def test_weight_compatibility_is_checked_on_reused_compiled_calls():
    left = JaxSample(jnp.ones((3, 4)), 1, jnp.arange(1.0, 5.0))
    add = jax.jit(checkify.checkify(operator.add))

    error, result = add(left, left)
    error.throw()
    np.testing.assert_array_equal(result.weights, left.weights)

    right = JaxSample(left.array, 1, jnp.ones(4))
    error, _ = add(left, right)
    with pytest.raises(ValueError, match="matching sample weights"):
        error.throw()


def test_grad_includes_array_and_weights():
    sample = JaxSample(jnp.arange(12.0).reshape(3, 4), 1, jnp.arange(1.0, 5.0))

    gradient = jax.jit(jax.grad(lambda value: value.sample_mean().sum()))(sample)

    assert isinstance(gradient, JaxSample)
    assert gradient.sample_axis == 1
    expected_array = jnp.broadcast_to(sample.weights / sample.weights.sum(), sample.shape)
    expected_weights = (sample.array - sample.sample_mean()[:, None]).sum(axis=0) / sample.weights.sum()
    np.testing.assert_allclose(gradient.array, expected_array, rtol=1e-6)
    np.testing.assert_allclose(gradient.weights, expected_weights, rtol=1e-6, atol=1e-6)


def test_stop_gradient_preserves_dynamic_fields():
    sample = JaxSample(jnp.ones((3, 4)), 1, jnp.ones(4))
    gradient = jax.grad(lambda value: value.stop_gradient().sample_mean().sum())(sample)

    assert gradient.sample_axis == 1
    np.testing.assert_array_equal(gradient.array, jnp.zeros_like(sample.array))
    np.testing.assert_array_equal(gradient.weights, jnp.zeros_like(sample.weights))


def test_nested_distribution_pytree_round_trip():
    distribution = JaxProbabilityCategoricalDistribution(jnp.ones((3, 4, 2)))
    sample = JaxSample(distribution, 1, jnp.arange(1.0, 5.0))

    result = jax.jit(lambda value: value)(sample)

    assert isinstance(result, JaxSample)
    assert isinstance(result.array, JaxProbabilityCategoricalDistribution)
    assert result.sample_axis == 1
    assert len(jax.tree.leaves(result)) == 2
    np.testing.assert_array_equal(result.array.probabilities, distribution.probabilities)
    np.testing.assert_array_equal(result.weights, sample.weights)


def test_opaque_tree_mapping_uses_sample_operations():
    sample = JaxSample(jnp.ones((3, 4)), 1, jnp.arange(1.0, 5.0))

    result = jax_tree_map(lambda value: value * 2, {"sample": sample})["sample"]

    assert isinstance(result, JaxSample)
    np.testing.assert_array_equal(result.array, sample.array * 2)
    np.testing.assert_array_equal(result.weights, sample.weights)


@pytest.mark.parametrize("method", ["add", "sin", "reduce", "accumulate", "outer", "modf"])
def test_numpy_ufuncs_keep_native_numpy_results(method):
    array = np.arange(1.0, 13.0, dtype=np.float32).reshape(3, 4)
    sample = JaxSample(jnp.asarray(array), 1, jnp.ones(4))

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
    sample = JaxSample(jnp.ones((3, 4)), 1, jnp.ones(4))
    array = np.full((3, 4), 2.0, dtype=np.float32)
    out = np.empty_like(array)

    assert np.add(sample, array, out=out) is out
    np.testing.assert_array_equal(out, array + 1)
    assert isinstance(array + sample, np.ndarray)
    np.testing.assert_array_equal(array + sample, array + 1)

    result = sample + array
    assert isinstance(result, JaxSample)
    np.testing.assert_array_equal(result.array, array + 1)
    np.testing.assert_array_equal(result.weights, sample.weights)
