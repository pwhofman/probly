"""Tests for generalized jax protected-axis behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.jax_functions import (
    jax_average,
    jax_concatenate,
    jax_copy,
    jax_expand_dims,
    jax_mean,
    jax_moveaxis,
    jax_squeeze,
    jax_stack,
    jax_sum,
    jax_take_along_axis,
    jax_transpose,
)


@dataclass(frozen=True, slots=True)
class SingleArray(JaxAxisProtected[Any]):
    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}


@dataclass(frozen=True, slots=True)
class ReductionArray(JaxAxisProtected[Any]):
    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Any]] = {jax_mean, jax_sum, jax_average}


@dataclass(frozen=True, slots=True)
class PairArray(JaxAxisProtected[Any]):
    first: jax.Array
    second: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"first": 0, "second": 0}


@dataclass(frozen=True, slots=True)
class InnerPairArray(JaxAxisProtected[Any]):
    left: jax.Array
    right: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}


@dataclass(frozen=True, slots=True)
class MixedArrayNumpy(JaxAxisProtected[Any]):
    array: jax.Array
    sidecar: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1, "sidecar": 1}


@dataclass(frozen=True, slots=True)
class MixedBatchArrayNumpy(JaxAxisProtected[Any]):
    array: jax.Array
    sidecar: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 0, "sidecar": 0}


@dataclass(frozen=True, slots=True)
class MixedReductionArrayNumpy(JaxAxisProtected[Any]):
    array: jax.Array
    sidecar: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1, "sidecar": 1}
    permitted_functions: ClassVar[set[Any]] = {jax_sum}


@dataclass(frozen=True, slots=True)
class OuterNestedArray(JaxAxisProtected[Any]):
    inner: InnerPairArray
    aux: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"inner": 1, "aux": 1}


def _mixed_array_numpy() -> MixedArrayNumpy:
    sidecar = np.asarray([f"v{i}" for i in range(24)], dtype=object).reshape(2, 3, 4)
    return MixedArrayNumpy(
        array=jnp.arange(24.0).reshape(2, 3, 4),
        sidecar=sidecar,
    )


def test_single_field_jax_functions_preserve_type() -> None:
    x = SingleArray(jnp.arange(6.0).reshape(2, 3))

    expanded = jax_expand_dims(x, axis=0)
    assert isinstance(expanded, SingleArray)
    assert tuple(expanded.array.shape) == (1, 2, 3)

    stacked = jax_stack((x, x), axis=0)
    assert isinstance(stacked, SingleArray)
    assert tuple(stacked.array.shape) == (2, 2, 3)


def test_squeeze_without_axis_drops_every_size_one_batch_axis() -> None:
    x = SingleArray(jnp.arange(24.0).reshape(1, 2, 1, 3, 4))

    squeezed = jax_squeeze(x)

    assert isinstance(squeezed, SingleArray)
    assert tuple(squeezed.array.shape) == (2, 3, 4)
    assert squeezed.shape == (2, 3)
    assert squeezed.protected_shape == (4,)


def test_squeeze_without_axis_keeps_size_one_protected_axis() -> None:
    """The trailing protected axis is never squeezed, even when it has size one."""
    x = SingleArray(jnp.arange(2.0).reshape(1, 2, 1))

    squeezed = jax_squeeze(x)

    assert isinstance(squeezed, SingleArray)
    assert tuple(squeezed.array.shape) == (2, 1)
    assert squeezed.protected_shape == (1,)


def test_squeeze_without_axis_is_a_no_op_without_size_one_batch_axes() -> None:
    x = SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    squeezed = jax_squeeze(x)

    assert isinstance(squeezed, SingleArray)
    assert tuple(squeezed.array.shape) == (2, 3, 4)


def test_reshape_method_preserves_protected_axes() -> None:
    x = SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    reshaped = x.reshape(6)
    assert isinstance(reshaped, SingleArray)
    assert tuple(reshaped.array.shape) == (6, 4)
    assert reshaped.shape == (6,)
    assert reshaped.protected_shape == (4,)

    reshaped_with_tuple = x.reshape((1, 6))
    assert isinstance(reshaped_with_tuple, SingleArray)
    assert tuple(reshaped_with_tuple.array.shape) == (1, 6, 4)
    assert reshaped_with_tuple.shape == (1, 6)
    assert reshaped_with_tuple.protected_shape == (4,)


def test_take_along_axis_preserves_protected_axes() -> None:
    x = SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    index = jnp.array([[2, 0], [1, 1]])

    gathered = jax_take_along_axis(x, axis=1, indices=index)

    assert isinstance(gathered, SingleArray)
    assert tuple(gathered.array.shape) == (2, 2, 4)
    assert gathered.shape == (2, 2)
    assert gathered.protected_shape == (4,)
    assert jnp.allclose(gathered.array, jnp.take_along_axis(x.array, index[..., None], axis=1))


def test_take_along_axis_supports_negative_axis() -> None:
    x = SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    index = jnp.array([[2, 0], [1, 1]])

    gathered = jax_take_along_axis(x, axis=-1, indices=index)

    assert isinstance(gathered, SingleArray)
    assert jnp.allclose(gathered.array, jnp.take_along_axis(x.array, index[..., None], axis=1))


def test_take_along_axis_applies_to_all_fields() -> None:
    x = InnerPairArray(
        left=jnp.arange(24.0).reshape(2, 3, 4),
        right=jnp.arange(24.0).reshape(2, 3, 4) + 100,
    )
    index = jnp.array([[2, 0], [1, 1]])
    broadcasted_index = jnp.broadcast_to(index[..., None], (2, 2, 4))

    gathered = jax_take_along_axis(x, axis=1, indices=index)

    assert isinstance(gathered, InnerPairArray)
    assert tuple(gathered.left.shape) == (2, 2, 4)
    assert tuple(gathered.right.shape) == (2, 2, 4)
    assert jnp.allclose(gathered.left, jax_take_along_axis(x.left, axis=1, indices=broadcasted_index))
    assert jnp.allclose(gathered.right, jax_take_along_axis(x.right, axis=1, indices=broadcasted_index))


def test_take_along_axis_rejects_protected_axis_dim() -> None:
    x = SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    index = jnp.zeros((2, 3), dtype=jnp.int32)

    with pytest.raises(ValueError, match="axis 2 is out of bounds"):
        jax_take_along_axis(x, axis=2, indices=index)


def test_take_along_axis_rejects_index_with_wrong_batch_ndim() -> None:
    x = SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    index = jnp.zeros((2, 3, 1), dtype=jnp.int32)

    with pytest.raises(ValueError, match="same ndim"):
        jax_take_along_axis(x, axis=1, indices=index)


def test_take_along_axis_rejects_numpy_sidecar() -> None:
    x = _mixed_array_numpy()
    index = jnp.array([[2, 0], [1, 1]])

    with pytest.raises(TypeError):
        jax_take_along_axis(x, axis=1, indices=index)


def test_unpermitted_jax_reductions_return_notimplemented() -> None:
    x = SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError):
        _ = jax_sum(x)

    with pytest.raises(TypeError):
        _ = jax_mean(x)


def test_permitted_jax_reductions_reduce_only_batch_axes() -> None:
    x = ReductionArray(jnp.arange(24.0).reshape(2, 3, 4))

    summed = jax_sum(x)
    assert isinstance(summed, ReductionArray)
    assert jnp.allclose(summed.array, jnp.sum(x.array, axis=(0, 1)))
    assert summed.shape == ()
    assert summed.protected_shape == (4,)

    meaned = jax_mean(x, axis=1)
    assert isinstance(meaned, ReductionArray)
    assert jnp.allclose(meaned.array, jnp.mean(x.array, axis=1))
    assert meaned.shape == (2,)
    assert meaned.protected_shape == (4,)


def test_permitted_jax_average_reduced_only_batch_axes() -> None:
    x = ReductionArray(jnp.arange(24.0).reshape(2, 3, 4))
    weights = jnp.array([0.25, 0.75])

    averaged = jax_average(x, axis=0, weights=weights)

    assert isinstance(averaged, ReductionArray)
    assert jnp.allclose(averaged.array, jax_average(x.array, axis=0, weights=weights))
    assert averaged.shape == (3,)
    assert averaged.protected_shape == (4,)


def test_permitted_jax_average_expands_batch_shaped_weights_before_protected_axis() -> None:
    x = ReductionArray(jnp.arange(24.0).reshape(2, 3, 4))
    weights = jnp.array([[1.0, 3.0, 1.0], [2.0, 1.0, 2.0]])

    averaged = jax_average(x, axis=-1, weights=weights)

    expected = jax_average(x.array, axis=1, weights=jnp.broadcast_to(weights[..., None], x.array.shape))
    assert isinstance(averaged, ReductionArray)
    assert jnp.allclose(averaged.array, expected)
    assert averaged.shape == (2,)
    assert averaged.protected_shape == (4,)


def test_unpermitted_jax_average_returns_notimplemented() -> None:
    x = SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="no implementation found for jax_average"):
        _ = jax_average(x, axis=0)


def test_jax_average_rejects_returned() -> None:
    """``returned=True`` yields a tuple, which has no protected-axis interpretation."""
    x = ReductionArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="returned = True is not supported"):
        _ = jax_average(x, axis=0, returned=True)


def test_jax_reductions_support_keepdims() -> None:
    x = ReductionArray(jnp.arange(24.0).reshape(2, 3, 4))

    summed = jax_sum(x, axis=0, keepdims=True)

    assert isinstance(summed, ReductionArray)
    assert tuple(summed.array.shape) == (1, 3, 4)
    assert summed.shape == (1, 3)
    assert summed.protected_shape == (4,)


def test_multi_field_array_conversion_and_cat() -> None:
    x = PairArray(jnp.ones((2, 2)), jnp.ones((2, 2)) * 2)

    with pytest.raises(TypeError, match="Cannot convert multi-field"):
        _ = np.asarray(x)

    cat = jax_concatenate((x, x), axis=0)
    assert isinstance(cat, PairArray)
    assert tuple(cat.first.shape) == (4, 2)
    assert tuple(cat.second.shape) == (4, 2)


def test_reshape_method_applies_to_all_field() -> None:
    x = PairArray(jnp.arange(6.0).reshape(2, 3), jnp.arange(6.0).reshape(2, 3) + 10)

    reshaped = x.reshape(3, 2)
    assert isinstance(reshaped, PairArray)
    assert tuple(reshaped.first.shape) == (3, 2)
    assert tuple(reshaped.second.shape) == (3, 2)


def test_mixed_numpy_sidecar_structural_unary_ops() -> None:
    x = _mixed_array_numpy()

    reshaped = x.reshape(3, 2)
    assert isinstance(reshaped, MixedArrayNumpy)
    assert tuple(reshaped.array.shape) == (3, 2, 4)
    np.testing.assert_array_equal(reshaped.sidecar, np.reshape(x.sidecar, (3, 2, 4)))

    transposed = jax_transpose(x)
    assert isinstance(transposed, MixedArrayNumpy)
    assert tuple(transposed.array.shape) == (3, 2, 4)
    np.testing.assert_array_equal(transposed.sidecar, np.transpose(x.sidecar, axes=(1, 0, 2)))

    permuted = jax_transpose(x, axes=(1, 0))
    assert isinstance(permuted, MixedArrayNumpy)
    assert tuple(permuted.array.shape) == (3, 2, 4)
    np.testing.assert_array_equal(permuted.sidecar, np.transpose(x.sidecar, axes=(1, 0, 2)))

    expanded = jax_expand_dims(x, axis=1)
    assert isinstance(expanded, MixedArrayNumpy)
    assert tuple(expanded.array.shape) == (2, 1, 3, 4)
    np.testing.assert_array_equal(expanded.sidecar, np.expand_dims(x.sidecar, axis=1))

    squeezed = jax_squeeze(expanded, axis=1)
    assert isinstance(squeezed, MixedArrayNumpy)
    assert tuple(squeezed.array.shape) == (2, 3, 4)
    np.testing.assert_array_equal(squeezed.sidecar, x.sidecar)

    not_squeezed = jax_squeeze(x, axis=0)
    assert isinstance(not_squeezed, MixedArrayNumpy)
    assert tuple(not_squeezed.array.shape) == (2, 3, 4)
    np.testing.assert_array_equal(not_squeezed.sidecar, x.sidecar)

    moved = jax_moveaxis(x, 0, 1)
    assert isinstance(moved, MixedArrayNumpy)
    assert tuple(moved.array.shape) == (3, 2, 4)
    np.testing.assert_array_equal(moved.sidecar, np.moveaxis(x.sidecar, 0, 1))

    matrix_transposed = x.mT
    assert isinstance(matrix_transposed, MixedArrayNumpy)
    assert tuple(matrix_transposed.array.shape) == (3, 2, 4)
    np.testing.assert_array_equal(matrix_transposed.sidecar, np.swapaxes(x.sidecar, 0, 1))

    adjointed = x.mH
    assert isinstance(adjointed, MixedArrayNumpy)
    assert tuple(adjointed.array.shape) == (3, 2, 4)
    np.testing.assert_array_equal(adjointed.sidecar, np.swapaxes(x.sidecar, 0, 1))


def test_mixed_numpy_sidecar_sequence_ops() -> None:
    x = _mixed_array_numpy()

    stacked = jax_stack((x, x), axis=0)
    assert isinstance(stacked, MixedArrayNumpy)
    assert tuple(stacked.array.shape) == (2, 2, 3, 4)
    np.testing.assert_array_equal(stacked.sidecar, np.stack((x.sidecar, x.sidecar), axis=0))

    cat = jax_concatenate((x, x), axis=1)
    assert isinstance(cat, MixedArrayNumpy)
    assert tuple(cat.array.shape) == (2, 6, 4)
    np.testing.assert_array_equal(cat.sidecar, np.concatenate((x.sidecar, x.sidecar), axis=1))


def test_mixed_numpy_sidecar_index_and_astype() -> None:
    x = MixedBatchArrayNumpy(
        array=jnp.array([1.0, 2.0]),
        sidecar=np.asarray(["a", "b"], dtype=object),
    )

    indexed = x[1]
    assert isinstance(indexed, MixedBatchArrayNumpy)
    assert tuple(indexed.array.shape) == ()
    assert indexed.sidecar.shape == ()
    assert indexed.sidecar.item() == "b"
    assert jnp.equal(indexed.array, jnp.array(2.0))

    converted = x.astype(jnp.float16)
    assert converted.sidecar is x.sidecar
    assert converted.array.dtype == jnp.float16


def test_mixed_numpy_sidecar_rejects_non_structural_jax_ops() -> None:
    x = _mixed_array_numpy()

    with pytest.raises(TypeError):
        _ = jax_copy(x)

    reducible = MixedReductionArrayNumpy(array=x.array, sidecar=x.sidecar)
    with pytest.raises(TypeError):
        _ = jax_sum(reducible)


def test_multi_field_index_and_assignment() -> None:
    x = PairArray(jnp.zeros((2, 2)), jnp.ones((2, 2)))

    y = x[0]
    assert isinstance(y, PairArray)
    assert tuple(y.first.shape) == (2,)
    assert tuple(y.second.shape) == (2,)

    x = x.at[0].set(PairArray(jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])))
    assert jnp.array_equal(x.first[0], jnp.array([3.0, 4.0]))
    assert jnp.array_equal(x.second[0], jnp.array([5.0, 6.0]))

    x = x.at[1].set((jnp.array([7.0, 8.0]), jnp.array([9.0, 10.0])))
    assert jnp.array_equal(x.first[1], jnp.array([7.0, 8.0]))
    assert jnp.array_equal(x.second[1], jnp.array([9.0, 10.0]))


def test_nested_multi_field_value_delegates_without_array_casts() -> None:
    inner = InnerPairArray(
        left=jnp.arange(24.0).reshape(2, 3, 4),
        right=jnp.arange(24.0).reshape(2, 3, 4) + 100,
    )
    outer = OuterNestedArray(inner=inner, aux=jnp.arange(10.0).reshape(2, 5))

    values = outer.protected_values()
    assert values["inner"] is inner

    expanded = jax_expand_dims(outer, axis=0)
    assert isinstance(expanded, OuterNestedArray)
    assert isinstance(expanded.inner, InnerPairArray)
    assert tuple(expanded.inner.left.shape) == (1, 2, 3, 4)
    assert tuple(expanded.inner.right.shape) == (1, 2, 3, 4)
    assert tuple(expanded.aux.shape) == (1, 2, 5)

    stacked = jax_stack((outer, outer), axis=0)
    assert isinstance(stacked, OuterNestedArray)
    assert isinstance(stacked.inner, InnerPairArray)
    assert tuple(stacked.inner.left.shape) == (2, 2, 3, 4)
    assert tuple(stacked.inner.right.shape) == (2, 2, 3, 4)
    assert tuple(stacked.aux.shape) == (2, 2, 5)

    sliced = outer[1]
    assert isinstance(sliced, OuterNestedArray)
    assert isinstance(sliced.inner, InnerPairArray)
    assert tuple(sliced.inner.left.shape) == (3, 4)
    assert tuple(sliced.inner.right.shape) == (3, 4)
    assert tuple(sliced.aux.shape) == (5,)
