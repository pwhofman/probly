"""Tests for the individual code paths of ``JaxAxisProtected``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.jax_functions import (
    jax_astype,
    jax_average,
    jax_concatenate,
    jax_copy,
    jax_expand_dims,
    jax_matrix_transpose,
    jax_mean,
    jax_moveaxis,
    jax_reshape,
    jax_squeeze,
    jax_stack,
    jax_std,
    jax_sum,
    jax_swapaxes,
    jax_take_along_axis,
    jax_transpose,
    jax_var,
)


@dataclass(frozen=True, slots=True)
class _SingleArray(JaxAxisProtected[Any]):
    """Single jax array field with one protected trailing axis."""

    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Any]] = {jax_mean, jax_sum, jax_average, jax_std, jax_var}


@dataclass(frozen=True, slots=True)
class _ScalarArray(JaxAxisProtected[Any]):
    """Single jax array field with no protected trailing axis."""

    array: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}


@dataclass(frozen=True, slots=True)
class _PairArray(JaxAxisProtected[Any]):
    """Two jax array field with no protected trailing axes."""

    first: jax.Array
    second: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"first": 0, "second": 0}


@dataclass(frozen=True, slots=True)
class _PairArrayProtected(JaxAxisProtected[Any]):
    """Two jax array field each with one protected trailing axis."""

    left: jax.Array
    right: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"left": 1, "right": 1}


@dataclass(frozen=True, slots=True)
class _NumpyOnly(JaxAxisProtected[Any]):
    """Edge case: a jax-protected representation whose only field is numpy.

    The representation is unusual but allowed; it triggers the ``_jax_protected_value``
    error path because no jax-like value is present.
    """

    sidecar: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"sidecar": 1}


@dataclass(frozen=True, slots=True)
class _MixedArrayNumpy(JaxAxisProtected[Any]):
    """A jax array alongside a numpy sidecar, both with one protected trailing axis."""

    array: jax.Array
    sidecar: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1, "sidecar": 1}


@dataclass(frozen=True, slots=True)
class _MixedBatchArrayNumpy(JaxAxisProtected[Any]):
    """A jax array alongside a numpy object sidecar, with no protected trailing axes."""

    array: jax.Array
    sidecar: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 0, "sidecar": 0}


# ---------------------------------------------------------------------------
# protected_values: permitted vs unpermitted func.
# ---------------------------------------------------------------------------


def test_jax_protected_values_no_args_returns_all_value() -> None:
    """``protected_values()`` without arguments returns all fields."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))
    values = x.protected_values()
    assert "array" in values
    assert jnp.allclose(values["array"], x.array)


def test_jax_protected_values_with_permitted_func() -> None:
    """``protected_values`` with a permitted function returns the dict."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))
    values = x.protected_values(jax_sum)
    assert values is not None


def test_jax_protected_values_with_unpermitted_func_returns_none() -> None:
    """``protected_values`` with an unpermitted function returns None."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))
    assert x.protected_values(jnp.cumsum) is None


# ---------------------------------------------------------------------------
# _jax_protected_value: must find at least one array field.
# ---------------------------------------------------------------------------


def test_jax_protected_value_raises_for_numpy_only_layout() -> None:
    """A jax-protected layout with only numpy fields raises on ``_jax_protected_value``."""
    x = _NumpyOnly(np.arange(6.0).reshape(2, 3))

    with pytest.raises(TypeError, match="No jax-like protected value"):
        _ = x._jax_protected_value()  # noqa: SLF001


# ---------------------------------------------------------------------------
# __len__ / __iter__ for zero-dim representations.
# ---------------------------------------------------------------------------


def test_len_raises_for_zero_dim_distribution() -> None:
    """``__len__`` raises on zero-dim jax protected representations."""
    x = _SingleArray(jnp.array([1.0, 2.0, 3.0]))
    assert x.shape == ()
    assert x.ndim == 0

    with pytest.raises(TypeError, match="unsized distribution"):
        _ = len(x)


def test_iter_yields_object() -> None:
    """Iteration walks the leading batch axis."""
    x = _SingleArray(jnp.arange(12.0).reshape(3, 4))
    items = list(x)
    assert len(items) == 3
    for i, item in enumerate(items):
        assert isinstance(item, _SingleArray)
        np.testing.assert_allclose(item.array, x.array[i])


# ---------------------------------------------------------------------------
# __array_namespace__, dtype, device.
# ---------------------------------------------------------------------------


def test_array_namespace_delegates_to_underlying_array() -> None:
    """``__array_namespace__`` delegates to the underlying jax array's implementation."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))
    assert x.__array_namespace__() is x.array.__array_namespace__()


def test_dtype_property_delegates_to_jax_value() -> None:
    """``dtype`` returns the underlying jax dtype."""
    x = _SingleArray(jnp.ones((2, 3), dtype=jnp.float16))
    assert x.dtype == jnp.float16


def test_device_property_delegates_to_jax_value() -> None:
    """``device`` returns the underlying jax device."""
    x = _SingleArray(jnp.ones((2, 3)))
    assert x.device == x.array.device


# ---------------------------------------------------------------------------
# size with int dim.
# ---------------------------------------------------------------------------


def test_size_int_dim_return_batch_size() -> None:
    """``size(int)`` returns the batch size at that dim."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    assert x.size(0) == 2
    assert x.size(1) == 3
    assert x.size(-1) == 3
    assert x.size() == 6


def test_size_int_dim_out_of_bounds_raises() -> None:
    """``size(dim)`` with out-of-range dim raises IndexError."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    with pytest.raises(IndexError, match="out of bounds"):
        _ = x.size(5)


# ---------------------------------------------------------------------------
# mT and mH on insufficient ndim.
# ---------------------------------------------------------------------------


def test_mT_requires_two_batch_dims() -> None:  # noqa: N802
    """``mT`` requires ndim >= 2."""
    x = _SingleArray(jnp.arange(8.0).reshape(2, 4))  # batch ndim = 1
    with pytest.raises(ValueError, match="at least 2 batch dimensions"):
        _ = x.mT


def test_mH_requires_two_batch_dims() -> None:  # noqa: N802
    """``mH`` requires ndim >= 2."""
    x = _SingleArray(jnp.arange(8.0).reshape(2, 4))  # batch ndim = 1
    with pytest.raises(ValueError, match="at least 2 batch dimensions"):
        _ = x.mH


# ---------------------------------------------------------------------------
# Indexing patterns.
# ---------------------------------------------------------------------------


def test_getitem_int() -> None:
    """Integer indexing into batch axis works."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    y = x[0]
    assert isinstance(y, _SingleArray)
    assert tuple[int, ...](y.array.shape) == (3, 4)


def test_getitem_slice() -> None:
    """Slice indexing into batch axis works."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    y = x[:1]
    assert isinstance(y, _SingleArray)
    assert tuple[int, ...](y.array.shape) == (1, 3, 4)


def test_getitem_ellipsis() -> None:
    """Ellipsis indexing returns the same shape representation."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    y = x[...]
    assert isinstance(y, _SingleArray)
    assert tuple[int, ...](y.array.shape) == (2, 3, 4)


def test_getitem_tuple() -> None:
    """Tuple indexing across batch axes work."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    y = x[0, 1]
    assert isinstance(y, _SingleArray)
    assert tuple[int, ...](y.array.shape) == (4,)


def test_getitem_axes_zero_promotes_scalar() -> None:
    """For ``axes=0``, scalar results are promoted (jax array)."""
    x = _ScalarArray(jnp.arange(6.0))
    y = x[0]
    assert isinstance(y, _ScalarArray)
    assert y.array.ndim == 0


# ---------------------------------------------------------------------------
# Setitem error paths.
# ---------------------------------------------------------------------------


def test_setitem_redirects_to_at() -> None:
    """In-place assignment is rejected because jax arrays are immutable."""
    x = _PairArray(jnp.zeros((2, 2)), jnp.ones((2, 2)))
    with pytest.raises(TypeError, match=r"immutable.*at\[index\]\.set"):
        x[0] = _PairArray(jnp.zeros(2), jnp.ones(2))


def test_at_set_rejects_tuple_with_wrong_length() -> None:
    """Assigning a tuple with the wrong number of fields raises TypeError."""
    x = _PairArray(jnp.zeros((2, 2)), jnp.ones((2, 2)))
    with pytest.raises(TypeError, match="Expected tuple"):
        _ = x.at[0].set((jnp.zeros(2),))


def test_at_set_rejects_scalar_for_multi_field() -> None:
    """Single value assignment is rejected for multi-field protected objects."""
    x = _PairArray(jnp.zeros((2, 2)), jnp.ones((2, 2)))
    with pytest.raises(TypeError, match="multi-field protected object"):
        _ = x.at[0].set(jnp.zeros(2))


def test_at_set_rejects_value_with_wrong_protected_shape() -> None:
    """Assigning a value whose protected trailing axes differ raises ValueError."""
    x = _SingleArray(jnp.zeros((2, 4)))
    bad = _SingleArray(jnp.zeros((1, 5)))  # protected size 5

    with pytest.raises(ValueError, match="modifies protected trailing axes"):
        _ = x.at[0].set(bad)


# ---------------------------------------------------------------------------
# to.
# ---------------------------------------------------------------------------


def test_astype_returns_self_when_already_correct() -> None:
    """``astype`` returns the original instance when no conversion needed."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))
    y = x.astype(dtype=x.dtype)
    assert y is x


def test_astype_changes_dtype() -> None:
    """``astype`` returns a new instance with converted dtype."""
    x = _SingleArray(jnp.ones((2, 3), dtype=jnp.float32))
    y = x.astype(dtype=jnp.float16)
    assert isinstance(y, _SingleArray)
    assert y.array.dtype == jnp.float16


# ---------------------------------------------------------------------------
# numpy and __array__.
# ---------------------------------------------------------------------------


def test_numpy_returns_array_for_single_field() -> None:
    """``numpy()`` on a single-field returns a numpy array."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))
    arr = x.numpy()
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.asarray(x.array))


def test_numpy_force_copies_data() -> None:
    """``numpy(force=True)`` copies the underlying buffer."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))
    arr = x.numpy(force=True)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.asarray(x.array))

    arr[0, 0] = 99.0
    assert float(x.array[0, 0]) == 0.0


def test_numpy_with_numpy_field_force_returns_copy() -> None:
    """Numpy fields are returned as a copy when ``force=True``."""
    x = _NumpyOnly(np.arange(6.0).reshape(2, 3))
    arr = x.numpy(force=True)
    assert isinstance(arr, np.ndarray)
    assert arr is not x.sidecar


def test_numpy_with_numpy_field_no_force_returns_same() -> None:
    """Numpy fields are returned as-is when ``force=False``."""
    x = _NumpyOnly(np.arange(6.0).reshape(2, 3))
    arr = x.numpy(force=False)
    assert arr is x.sidecar


def test_numpy_rejects_multi_field() -> None:
    """``numpy()`` is not defined for multi-field representations."""
    x = _PairArray(jnp.zeros((2, 3)), jnp.ones((2, 3)))
    with pytest.raises(TypeError, match="multi-field"):
        _ = x.numpy()


def test_array_dunder_returns_ndarray() -> None:
    """``np.asarray`` returns an ndarray with proper dtype."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))
    arr = np.asarray(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr, np.asarray(x.array))


def test_array_dunder_with_dtype_converts() -> None:
    """``np.asarray(..., dtype=...)`` converts the dtype."""
    x = _SingleArray(jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3))
    arr = np.asarray(x, dtype=np.float16)
    assert arr.dtype == np.float16


def test_array_dunder_with_copy_true_returns_copy() -> None:
    """``np.asarray(..., copy=True)`` returns a freshly allocated array."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))
    arr = np.array(x, copy=True)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.asarray(x.array))

    arr[0, 0] = 99.0
    assert float(x.array[0, 0]) == 0.0


# ---------------------------------------------------------------------------
# reshape and torch_function dispatch.
# ---------------------------------------------------------------------------


def test_reshape_method_uses_jax_function() -> None:
    """``self.reshape(...)`` is dispatched via jax_function."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    y = x.reshape(6)
    assert isinstance(y, _SingleArray)
    assert tuple(y.array.shape) == (6, 4)


def test_jax_function_classmethod_dispatches() -> None:
    """The class-level ``__jax_function__`` is the entry point for jax ops."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    y = jax_swapaxes(x, 0, 1)
    assert isinstance(y, _SingleArray)
    assert tuple(y.array.shape) == (3, 2, 4)


# ---------------------------------------------------------------------------
# at[...] indexed updates.
# ---------------------------------------------------------------------------


def test_at_get_returns_indexed_slice() -> None:
    """``at[index].get()`` mirrors ``__getitem__`` over the batch dimensions."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    y = x.at[0].get()

    assert isinstance(y, _SingleArray)
    assert tuple(y.array.shape) == (3, 4)
    assert jnp.array_equal(y.array, x.array[0])


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("add", jnp.array([3.0, 4.0])),
        ("subtract", jnp.array([-1.0, 0.0])),
        ("multiply", jnp.array([2.0, 4.0])),
        ("divide", jnp.array([0.5, 1.0])),
        ("power", jnp.array([1.0, 4.0])),
        ("min", jnp.array([1.0, 2.0])),
        ("max", jnp.array([2.0, 2.0])),
    ],
)
def test_at_arithmetic_updates_apply_to_batch_entry(method: str, expected: jax.Array) -> None:
    """Every ``jax.Array.at`` arithmetic update is forwarded to the protected fields."""
    x = _SingleArray(jnp.array([[1.0, 2.0], [5.0, 6.0]]))

    y = getattr(x.at[0], method)(2.0)

    assert isinstance(y, _SingleArray)
    assert jnp.allclose(y.array[0], expected)
    assert jnp.allclose(y.array[1], jnp.array([5.0, 6.0]))
    assert jnp.allclose(x.array[0], jnp.array([1.0, 2.0]))


def test_at_set_accepts_bare_value_for_single_field() -> None:
    """A single-field object accepts a bare replacement value."""
    x = _SingleArray(jnp.zeros((2, 4)))

    y = x.at[1].set(jnp.arange(4.0))

    assert jnp.allclose(y.array[1], jnp.arange(4.0))
    assert jnp.allclose(y.array[0], jnp.zeros(4))


def test_at_arithmetic_update_rejects_numpy_sidecar() -> None:
    """Arithmetic updates have no meaning for numpy sidecars and are rejected."""
    x = _MixedArrayNumpy(
        array=jnp.arange(24.0).reshape(2, 3, 4),
        sidecar=np.asarray([f"v{i}" for i in range(24)], dtype=object).reshape(2, 3, 4),
    )

    with pytest.raises(TypeError, match="not supported for NumPy field"):
        _ = x.at[0].add((1.0, 1.0))


def test_at_set_unwraps_zero_dim_object_array() -> None:
    """A 0-d object ndarray is unwrapped so it is not nested inside the sidecar."""
    x = _MixedBatchArrayNumpy(
        array=jnp.array([1.0, 2.0]),
        sidecar=np.asarray(["a", "b"], dtype=object),
    )

    y = x.at[0].set(x[1])

    assert y.sidecar[0] == "b"
    assert not isinstance(y.sidecar[0], np.ndarray)
    assert float(y.array[0]) == 2.0


# ---------------------------------------------------------------------------
# Function dispatch paths not reached through the convenience methods.
# ---------------------------------------------------------------------------


def test_matrix_transpose_swaps_trailing_batch_axes() -> None:
    """``jax_matrix_transpose`` transposes the last two batch axes only."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    y = jax_matrix_transpose(x)

    assert isinstance(y, _SingleArray)
    assert tuple(y.array.shape) == (3, 2, 4)
    assert y.protected_shape == (4,)
    assert jnp.array_equal(y.array, jnp.swapaxes(x.array, 0, 1))


def test_matrix_transpose_requires_two_batch_dims() -> None:
    """``jax_matrix_transpose`` needs at least two batch dimensions."""
    x = _SingleArray(jnp.arange(8.0).reshape(2, 4))

    with pytest.raises(ValueError, match="at least 2 batch dimensions"):
        _ = jax_matrix_transpose(x)


def test_astype_function_dispatches_to_protected_fields() -> None:
    """``jax_astype`` goes through the dispatch layer, unlike the ``astype`` method."""
    x = _SingleArray(jnp.ones((2, 3), dtype=jnp.float32))

    y = jax_astype(x, jnp.float16, copy=True, device=None)

    assert isinstance(y, _SingleArray)
    assert y.array.dtype == jnp.float16


def test_copy_function_copies_single_field() -> None:
    """``jax_copy`` returns an equal but distinct single-field object."""
    x = _SingleArray(jnp.arange(6.0).reshape(2, 3))

    y = jax_copy(x)

    assert isinstance(y, _SingleArray)
    assert y is not x
    assert jnp.array_equal(y.array, x.array)


def test_copy_function_copies_every_field() -> None:
    """``jax_copy`` applies to all fields."""
    x = _PairArrayProtected(jnp.zeros((2, 3)), jnp.ones((2, 3)))

    y = jax_copy(x)

    assert isinstance(y, _PairArrayProtected)
    assert jnp.array_equal(y.left, x.left)
    assert jnp.array_equal(y.right, x.right)


def test_copy_function_declines_numpy_sidecars() -> None:
    """A numpy sidecar cannot survive ``jnp.copy``, so the handler declines."""
    x = _MixedArrayNumpy(
        array=jnp.arange(24.0).reshape(2, 3, 4),
        sidecar=np.asarray([f"v{i}" for i in range(24)], dtype=object).reshape(2, 3, 4),
    )

    with pytest.raises(TypeError, match="no implementation found for jax_copy"):
        _ = jax_copy(x)


@pytest.mark.parametrize("func", [jax_std, jax_var])
def test_std_and_var_reduce_only_batch_axes(func: Callable[..., Any]) -> None:
    """``jax_std``/``jax_var`` are registered alongside the other batch reductions."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    y = func(x, axis=0)

    assert isinstance(y, _SingleArray)
    assert y.shape == (3,)
    assert y.protected_shape == (4,)
    assert jnp.allclose(y.array, func(x.array, axis=0))


def test_take_along_axis_forwards_mode() -> None:
    """``mode`` is forwarded to ``jnp.take_along_axis``."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    indices = jnp.array([[9, 0], [1, 1]])

    y = jax_take_along_axis(x, indices, axis=1, mode="clip")

    assert isinstance(y, _SingleArray)
    assert tuple(y.array.shape) == (2, 2, 4)
    assert jnp.array_equal(y.array[0, 0], x.array[0, 2])


def test_moveaxis_accepts_sequence_source_and_destination() -> None:
    """Tuple ``source``/``destination`` are normalized against the batch dimensions."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    y = jax_moveaxis(x, (0,), (1,))

    assert isinstance(y, _SingleArray)
    assert tuple(y.array.shape) == (3, 2, 4)
    assert y.protected_shape == (4,)


def test_reshape_function_expands_none_entries() -> None:
    """``None`` entries in the target shape are treated as size one."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    y = jax_reshape(x, (None, 6))

    assert isinstance(y, _SingleArray)
    assert tuple(y.array.shape) == (1, 6, 4)


def test_jax_like_converts_dtype() -> None:
    """``__jax_like__`` delegates to ``astype``."""
    x = _SingleArray(jnp.ones((2, 3), dtype=jnp.float32))

    y = x.__jax_like__(jnp.float16)

    assert isinstance(y, _SingleArray)
    assert y.array.dtype == jnp.float16


def test_with_protected_values_keeps_unlisted_fields() -> None:
    """Fields absent from the update dict keep their current value."""
    x = _PairArrayProtected(jnp.zeros((2, 4)), jnp.ones((2, 4)))

    y = x.with_protected_values({"left": jnp.full((2, 4), 7.0)})

    assert isinstance(y, _PairArrayProtected)
    assert jnp.allclose(y.left, 7.0)
    assert y.right is x.right


# ---------------------------------------------------------------------------
# Error paths.
# ---------------------------------------------------------------------------


def test_protected_values_rejects_desynced_batch_shapes() -> None:
    """Protected fields must agree on their batch shape."""
    x = _PairArrayProtected(jnp.ones((2, 4)), jnp.ones((3, 4)))

    with pytest.raises(ValueError, match="do not share the same batch-shape"):
        _ = x.protected_values()


def test_concatenate_rejects_axis_none() -> None:
    """``axis=None`` would flatten the protected trailing axes into the batch."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(ValueError, match="modified protected trailing axes"):
        _ = jax_concatenate((x, x), axis=None)


def test_concatenate_rejects_non_int_axis() -> None:
    """A non-integer, non-None axis is rejected outright."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="concatenate axis must be an int or None"):
        _ = jax_concatenate((x, x), axis="0")


def test_stack_rejects_non_int_axis() -> None:
    """``stack`` requires an integer axis."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="stack axis must be an int"):
        _ = jax_stack((x, x), axis=None)


def test_sequence_ops_reject_mismatched_protected_axes() -> None:
    """Every protected input must declare the same protected_axes layout."""
    protected = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))
    unprotected = _ScalarArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(ValueError, match="identical protected_axes definitions"):
        _ = jax_stack((protected, unprotected), axis=0)


def test_sequence_ops_accept_raw_arrays_alongside_protected_inputs() -> None:
    """Raw arrays are passed through to the primary field of the protected template."""
    x = _SingleArray(jnp.ones((2, 4)))

    y = jax_concatenate((x, jnp.zeros((3, 4))), axis=0)

    assert isinstance(y, _SingleArray)
    assert tuple(y.array.shape) == (5, 4)
    assert y.protected_shape == (4,)


def test_reshape_rejects_non_shape_argument() -> None:
    """``reshape`` only accepts an int, tuple or list shape."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="reshape shape must be an int, tuple or list"):
        _ = jax_reshape(x, "bad")


def test_expand_dims_rejects_non_int_axis() -> None:
    """``expand_dims`` only accepts an int or a sequence of ints."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="expand_dims axis must be an int"):
        _ = jax_expand_dims(x, axis="0")


def test_squeeze_rejects_non_int_axis() -> None:
    """``squeeze`` only accepts an int or a sequence of ints."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="squeeze axis must be an int"):
        _ = jax_squeeze(x, axis="0")


def test_swapaxes_rejects_non_int_axis() -> None:
    """``swapaxes`` only accepts integer axes."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="swapaxes axis values must be integers"):
        _ = jax_swapaxes(x, "0", 1)


def test_moveaxis_rejects_non_int_source() -> None:
    """``moveaxis`` only accepts an int or a sequence of ints."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="moveaxis source must be an int"):
        _ = jax_moveaxis(x, "0", 1)


def test_moveaxis_rejects_non_int_destination() -> None:
    """``moveaxis`` only accepts an int or a sequence of ints as destination."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="moveaxis destination must be an int"):
        _ = jax_moveaxis(x, 0, "1")


def test_transpose_rejects_axes_of_wrong_length() -> None:
    """``transpose`` axes must cover exactly the batch dimensions."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(ValueError, match="must only refer to batch dimension"):
        _ = jax_transpose(x, axes=(0, 1, 2))


def test_transpose_rejects_non_int_axes() -> None:
    """``transpose`` axes must be integers."""
    x = _SingleArray(jnp.arange(24.0).reshape(2, 3, 4))

    with pytest.raises(TypeError, match="transpose axes must be a tuple/list of integers"):
        _ = jax_transpose(x, axes=("0", "1"))


def test_subclassing_a_concrete_protected_class_is_rejected() -> None:
    """``protected_axes`` is validated against the subclass' own annotations only."""
    with pytest.raises(TypeError, match="unknown field"):

        @dataclass(frozen=True, slots=True)
        class _Sub(_SingleArray):
            pass
