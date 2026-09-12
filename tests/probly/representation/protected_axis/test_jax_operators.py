"""Permission-controlled infix operations, broadcasting, and JAX transformations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import operator
from typing import ClassVar, Never

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from probly.representation import jax_functions as jf
from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.distribution.jax_dirichlet import JaxDirichletDistribution
from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution

_BINARY = [
    (operator.add, jf.jax_add, jnp.add),
    (operator.sub, jf.jax_subtract, jnp.subtract),
    (operator.mul, jf.jax_multiply, jnp.multiply),
    (operator.truediv, jf.jax_true_divide, jnp.true_divide),
    (operator.floordiv, jf.jax_floor_divide, jnp.floor_divide),
    (operator.mod, jf.jax_remainder, jnp.remainder),
    (operator.pow, jf.jax_power, jnp.power),
    (operator.and_, jf.jax_bitwise_and, jnp.bitwise_and),
    (operator.or_, jf.jax_bitwise_or, jnp.bitwise_or),
    (operator.xor, jf.jax_bitwise_xor, jnp.bitwise_xor),
    (operator.lshift, jf.jax_left_shift, jnp.left_shift),
    (operator.rshift, jf.jax_right_shift, jnp.right_shift),
]
_UNARY = [
    (operator.pos, jf.jax_positive, jnp.positive),
    (operator.neg, jf.jax_negative, jnp.negative),
    (operator.abs, jf.jax_absolute, jnp.absolute),
    (operator.invert, jf.jax_invert, jnp.invert),
]
_COMPARISONS = [
    (operator.eq, jf.jax_equal, jnp.equal),
    (operator.ne, jf.jax_not_equal, jnp.not_equal),
    (operator.lt, jf.jax_less, jnp.less),
    (operator.le, jf.jax_less_equal, jnp.less_equal),
    (operator.gt, jf.jax_greater, jnp.greater),
    (operator.ge, jf.jax_greater_equal, jnp.greater_equal),
]
_PERMITTED = {wrapper for _, wrapper, _ in [*_BINARY, *_UNARY, *_COMPARISONS]} | {jf.jax_divmod, jf.jax_matmul}


@dataclass(frozen=True, eq=False)
class Numeric(JaxAxisProtected[jax.Array]):
    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable]] = _PERMITTED


@dataclass(frozen=True, eq=False)
class Denied(JaxAxisProtected[jax.Array]):
    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}


@dataclass(frozen=True, eq=False)
class Pair(JaxAxisProtected[jax.Array]):
    first: jax.Array
    second: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"first": 1, "second": 2}
    permitted_functions: ClassVar[set[Callable]] = _PERMITTED


@pytest.mark.parametrize(("op", "wrapper", "backend"), _BINARY)
@pytest.mark.parametrize("kind", ["scalar", "jax", "numpy", "protected"])
def test_binary_operators_and_reflections(op, wrapper, backend, kind):
    x = Numeric(jnp.array([[2, 3], [4, 5]]))
    other = {
        "scalar": 2,
        "jax": jnp.array([1, 2]),
        "numpy": np.array([1, 2]),
        "protected": Numeric(jnp.array([1, 2])),
    }[kind]
    raw = other.array if isinstance(other, Numeric) else other
    for left, right, expected in (
        (x, other, backend(x.array, raw)),
        (other, x, backend(raw, x.array)),
    ):
        result = op(left, right)
        assert isinstance(result, Numeric)
        np.testing.assert_allclose(result.array, expected)
        np.testing.assert_allclose(wrapper(left, right).array, expected)
    np.testing.assert_array_equal(x.array, [[2, 3], [4, 5]])


@pytest.mark.parametrize(("op", "wrapper", "backend"), _UNARY)
def test_unary_operators(op, wrapper, backend):
    x = Numeric(jnp.array([[-2, 3], [4, -5]]))
    np.testing.assert_array_equal(op(x).array, backend(x.array))
    np.testing.assert_array_equal(wrapper(x).array, backend(x.array))
    with pytest.raises(TypeError):
        op(Denied(x.array))


@pytest.mark.parametrize(("op", "wrapper", "backend"), _BINARY)
@pytest.mark.parametrize("other", [2, np.array([1, 2]), jnp.array([1, 2])])
def test_arithmetic_denied_from_both_sides(op, wrapper, backend, other):
    del backend
    x = Denied(jnp.ones((2, 2), dtype=int))
    for left, right in ((x, other), (other, x)):
        with pytest.raises(TypeError):
            op(left, right)
        with pytest.raises(TypeError, match="no implementation found"):
            wrapper(left, right)


@pytest.mark.parametrize(("op", "wrapper", "backend"), _COMPARISONS)
def test_comparisons_return_batch_masks(op, wrapper, backend):
    x = Numeric(jnp.array([[1, 2], [3, 4], [2, 2]]))
    reduce = jnp.any if wrapper is jf.jax_not_equal else jnp.all
    expected = reduce(backend(x.array, 2), axis=-1)
    result = op(x, 2)
    assert isinstance(result, jax.Array)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(wrapper(x, 2), expected)
    with pytest.raises(TypeError):
        wrapper(Denied(x.array), 2)


def test_multifield_operations_and_comparisons():
    x = Pair(jnp.array([[1, 2], [1, 2]]), jnp.array([[[3, 4]], [[3, 4]]]))
    shifted = x + 2
    np.testing.assert_array_equal(shifted.first, x.first + 2)
    np.testing.assert_array_equal(shifted.second, x.second + 2)
    y = Pair(x.first, x.second.at[1, 0, 1].set(9))
    np.testing.assert_array_equal(x == y, [True, False])
    np.testing.assert_array_equal(x != y, [False, True])


def test_batch_broadcasting_preserves_protected_dimensions():
    x = Numeric(jnp.ones((2, 1, 3)))
    y = Numeric(jnp.arange(12).reshape(4, 3))
    result = x + y
    assert result.shape == (2, 4)
    assert result.protected_shape == (3,)
    np.testing.assert_array_equal(result.array, x.array + y.array)


def test_incompatible_protected_shapes_are_not_broadcast():
    with pytest.raises(ValueError, match="identical protected trailing shapes"):
        _ = Numeric(jnp.ones((2, 1))) + Numeric(jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="modified protected trailing axes"):
        _ = Numeric(jnp.ones((2, 1))) + jnp.ones((2, 3))


def test_unsynchronized_fields_are_rejected_after_broadcast():
    x = Pair(jnp.ones((1, 3)), jnp.ones((1, 2, 3)))
    with pytest.raises(ValueError, match="inconsistent batch-shapes"):
        _ = x + jnp.ones((2, 3))


def test_both_operands_must_permit_operation():
    @dataclass(frozen=True, eq=False)
    class Restricted(Numeric):
        array: jax.Array
        permitted_functions: ClassVar[set[Callable]] = set()

    x = Numeric(jnp.ones((2, 3)))
    y = Restricted(x.array)
    for args in ((x, y), (y, x)):
        with pytest.raises(TypeError):
            jf.jax_add(*args)
    # Even a subclass with identical storage must not silently choose a result type.
    with pytest.raises(TypeError):
        _ = x + Pair(x.array, x.array[..., None])


@pytest.mark.parametrize("dtype", [float, object])
def test_numpy_sidecars_are_rejected(dtype):
    @dataclass(frozen=True, eq=False)
    class Sidecar(JaxAxisProtected[jax.Array]):
        array: jax.Array
        sidecar: np.ndarray
        protected_axes: ClassVar[dict[str, int]] = {"array": 1, "sidecar": 0}
        permitted_functions: ClassVar[set[Callable]] = {jf.jax_add}

    x = Sidecar(jnp.ones((2, 3)), np.ones(2, dtype=dtype))
    with pytest.raises(TypeError):
        _ = x + 1


def test_python_reflected_fallback_and_implementation_errors():
    class Reflected:
        def __radd__(self, other) -> tuple[str, object]:
            return ("reflected", other)

    x = Numeric(jnp.ones((2, 3)))
    result = x + Reflected()
    assert result[0] == "reflected"
    assert result[1] is x

    class BrokenOverride:
        @classmethod
        def __jax_function__(cls, func, types, args, kwargs) -> Never:
            del cls, func, types, args, kwargs
            msg = "implementation error"
            raise TypeError(msg)

    with pytest.raises(TypeError, match="implementation error"):
        _ = x + BrokenOverride()


def test_divmod_reconstructs_both_results():
    x = Numeric(jnp.array([[5, 7], [8, 9]]))
    for left, right, raw_left, raw_right in ((x, 3, x.array, 3), (20, x, 20, x.array)):
        quotient, remainder = divmod(left, right)
        assert isinstance(quotient, Numeric)
        assert isinstance(remainder, Numeric)
        expected = jnp.divmod(raw_left, raw_right)
        np.testing.assert_array_equal(quotient.array, expected[0])
        np.testing.assert_array_equal(remainder.array, expected[1])


@pytest.mark.parametrize("left_shape", [(3,), (2, 3), (4, 2, 3)])
@pytest.mark.parametrize("right_shape", [(3,), (3, 5), (4, 3, 5)])
def test_matmul_operates_on_visible_dimensions(left_shape, right_shape):
    left = Numeric(jnp.arange(np.prod(left_shape) * 2).reshape((*left_shape, 2)))
    right = Numeric(jnp.arange(np.prod(right_shape) * 2).reshape((*right_shape, 2)))
    result = left @ right
    expected = jnp.stack([left.array[..., i] @ right.array[..., i] for i in range(2)], axis=-1)
    np.testing.assert_array_equal(result.array, expected)
    assert result.protected_shape == (2,)


def test_matmul_raw_operands_and_multiple_protected_axes():
    x = Pair(jnp.ones((2, 3, 4)), jnp.ones((2, 3, 4, 5)))
    right = jnp.ones((3, 6))
    result = x @ right
    assert result.first.shape == (2, 6, 4)
    assert result.second.shape == (2, 6, 4, 5)
    np.testing.assert_array_equal(result.second, jnp.full((2, 6, 4, 5), 3))
    result = jnp.ones((7, 2)) @ x
    assert isinstance(result, Pair)
    assert result.shape == (7, 3)
    np.testing.assert_array_equal(result.second, jnp.full((7, 3, 4, 5), 2))
    with pytest.raises(ValueError, match="ndim"):
        _ = Numeric(jnp.ones(4)) @ Numeric(jnp.ones(4))


def test_augmented_assignment_rebinds_without_mutating():
    x = Numeric(jnp.ones((2, 3)))
    original = x
    x += 2
    assert x is not original
    np.testing.assert_array_equal(original.array, jnp.ones((2, 3)))
    np.testing.assert_array_equal(x.array, jnp.full((2, 3), 3))
    with pytest.raises(TypeError):
        pow(x, 2, 3)


def test_result_hook_receives_context_and_cannot_change_protected_shape():
    @dataclass(frozen=True, eq=False)
    class BrokenResult(JaxAxisProtected[jax.Array]):
        array: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}
        permitted_functions: ClassVar[set[Callable]] = {jf.jax_subtract}

        def _postprocess_elementwise_result(self, values, *, func, operands):
            assert func is jf.jax_subtract
            assert operands[0] == 5
            assert operands[1] is self
            return {"array": values["array"][..., :1]}

    with pytest.raises(ValueError, match="modified protected trailing axes"):
        _ = 5 - BrokenResult(jnp.ones((2, 3)))


def test_traced_arithmetic_and_gradients():
    x = Numeric(jnp.arange(6.0).reshape(2, 3))
    np.testing.assert_array_equal(jax.jit(lambda a, b: a * b + 1)(x, 2).array, x.array * 2 + 1)
    np.testing.assert_array_equal(jax.vmap(lambda a: a + 1)(x).array, x.array + 1)
    gradient = jax.grad(lambda a: jnp.sum((a * a).array))(x)
    np.testing.assert_array_equal(gradient.array, 2 * x.array)


def test_distribution_opt_ins_and_traced_reconstruction():
    d = JaxDirichletDistribution(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    result = jax.jit(lambda x: 2 - x)(d)
    np.testing.assert_allclose(result.alphas, jnp.maximum(2 - d.alphas, 1e-10))
    with pytest.raises(TypeError):
        _ = d * 2
    g = JaxGaussianDistribution(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    shifted = jax.jit(lambda x, offset: offset - x)(g, 5)
    np.testing.assert_array_equal(shifted.mean, 5 - g.mean)
    np.testing.assert_array_equal(shifted.var, g.var)
    difference = g - g
    np.testing.assert_array_equal(difference.mean, jnp.zeros(2))
    np.testing.assert_array_equal(difference.var, 2 * g.var)
    with pytest.raises(TypeError):
        _ = g * 2


def test_eager_distribution_validation_is_preserved():
    with pytest.raises(ValueError, match="strictly positive"):
        JaxDirichletDistribution(jnp.array([-1.0, 2.0]))
    with pytest.raises(ValueError, match="Variance must be positive"):
        JaxGaussianDistribution(jnp.zeros(2), -jnp.ones(2))


def test_instance_permission_veto_is_respected_on_either_operand():
    @dataclass(frozen=True, eq=False)
    class Veto(Numeric):
        array: jax.Array
        deny: bool = False

        def protected_values(self, func=None):
            if self.deny and func is not None:
                return None
            return super().protected_values(func)

    allowed = Veto(jnp.ones((2, 3)))
    denied = Veto(allowed.array, deny=True)
    for left, right in ((allowed, denied), (denied, allowed)):
        with pytest.raises(TypeError):
            _ = left + right


def test_operation_context_reaches_reconstruction():
    @dataclass(frozen=True, eq=False)
    class Recorded(Numeric):
        array: jax.Array

        def with_protected_values(self, values, func=None):
            assert func is jf.jax_multiply
            return Recorded(values["array"])

    result = Recorded(jnp.ones((2, 3))) * 3
    np.testing.assert_array_equal(result.array, jnp.full((2, 3), 3))


def test_numpy_ufuncs_cannot_bypass_permissions_by_coercion():
    x = Denied(jnp.ones((2, 3)))
    with pytest.raises(TypeError):
        np.add(x, 1)
    np.testing.assert_array_equal(np.asarray(x), x.array)


@pytest.mark.parametrize(("op", "wrapper", "backend"), [*_BINARY, *_COMPARISONS])
def test_binary_wrappers_on_native_arrays(op, wrapper, backend):
    del op
    left, right = jnp.array([2, 4]), jnp.array([1, 2])
    np.testing.assert_array_equal(wrapper(left, right), backend(left, right))


@pytest.mark.parametrize(("op", "wrapper", "backend"), _UNARY)
def test_unary_wrappers_on_native_arrays(op, wrapper, backend):
    del op
    value = jnp.array([-2, 4])
    np.testing.assert_array_equal(wrapper(value), backend(value))
