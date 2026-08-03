"""Tests for the JAX-like protocols, base class and override machinery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from probly.representation.jax_functions import (
    handle_jax_function,
    has_jax_function,
    jax_average,
    jax_concatenate,
    jax_mean,
    jax_reshape,
    jax_stack,
    jax_std,
    jax_sum,
    jax_var,
)
from probly.representation.jax_like import (
    JaxLike,
    JaxLikeConvertible,
    JaxLikeImplementation,
    to_jax_like,
)


def _unwrap(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, _Wrapper):
        return value.array
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(item) for item in value)
    return value


@dataclass(frozen=True)
class _Wrapper(JaxLikeImplementation[jax.Array]):
    """Minimal concrete subclass that forwards every call to the wrapped array."""

    array: jax.Array
    label: str = "wrapped"

    @classmethod
    def __jax_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN206
        del types
        kwargs = {} if kwargs is None else kwargs
        result = func(*(_unwrap(arg) for arg in args), **{key: _unwrap(value) for key, value in kwargs.items()})
        return cls(result) if isinstance(result, jax.Array) else result

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def device(self):
        return self.array.device

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def size(self) -> int:
        return self.array.size


@dataclass(frozen=True)
class _Convertible:
    """Minimal ``JaxLikeConvertible`` used to check the ``to_jax_like`` registration."""

    values: list[float]

    def __jax_like__(self, dtype=None, /, *, device=None, copy=False):  # noqa: ANN204
        return jnp.asarray(self.values, dtype=dtype, device=device, copy=copy)

    def __array__(self, dtype=None, /, *, copy=None):  # noqa: ANN204
        return np.asarray(self.values, dtype=dtype, copy=copy)

    def __getitem__(self, index):  # noqa: ANN204
        return self.values[index]

    def __setitem__(self, index, value) -> None:
        self.values[index] = value

    def __array_namespace__(self, /, *, api_version=None):  # noqa: ANN204
        del api_version
        return np

    def to_device(self, device, /, *, stream=None):
        del device, stream
        return self

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):  # noqa: ANN204
        return iter(self.values)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.values),)

    @property
    def dtype(self):
        return np.float64

    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def T(self):  # noqa: N802
        return self

    @property
    def mT(self):  # noqa: N802
        return self


class TestProtocols:
    """The protocols must accept plain JAX arrays."""

    def test_jax_array_is_jax_like(self) -> None:
        assert isinstance(jnp.ones(3), JaxLike)

    def test_jax_array_is_jax_like_2d(self) -> None:
        assert isinstance(jnp.ones((2, 3)), JaxLike)

    def test_jax_array_is_jax_like_implementation(self) -> None:
        assert isinstance(jnp.ones(3), JaxLikeImplementation)

    def test_jax_array_is_not_convertible(self) -> None:
        assert not isinstance(jnp.ones(3), JaxLikeConvertible)

    def test_convertible_is_convertible(self) -> None:
        assert isinstance(_Convertible([1.0, 2.0]), JaxLikeConvertible)

    def test_jax_function_is_not_part_of_the_protocol(self) -> None:
        """``jax.Array`` has no ``__jax_function__``, so the protocol must not require it."""
        assert not hasattr(jnp.ones(3), "__jax_function__")
        assert "__jax_function__" not in JaxLike.__protocol_attrs__


class TestToJaxLike:
    """The ``to_jax_like`` dispatch."""

    def test_from_list(self) -> None:
        result = to_jax_like([1.0, 2.0, 3.0])
        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(np.asarray(result), [1.0, 2.0, 3.0])

    def test_from_numpy(self) -> None:
        result = to_jax_like(np.arange(4.0))
        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(np.asarray(result), np.arange(4.0))

    def test_dtype_is_honoured(self) -> None:
        result = to_jax_like([1, 2, 3], jnp.float32)
        assert result.dtype == jnp.float32

    def test_device_is_honoured(self) -> None:
        device = jax.devices()[0]
        result = to_jax_like([1, 2, 3], device=device)
        assert device in result.devices()

    def test_no_copy_returns_same_array(self) -> None:
        array = jnp.ones(3)
        assert to_jax_like(array) is array

    def test_copy_returns_new_array(self) -> None:
        array = jnp.ones(3)
        assert to_jax_like(array, copy=True) is not array

    def test_convertible_uses_dunder(self) -> None:
        result = to_jax_like(_Convertible([1.0, 2.0]), jnp.float32)
        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(result), [1.0, 2.0])


class TestOverrideMachinery:
    """``has_jax_function`` and ``handle_jax_function``."""

    def test_has_jax_function_false_for_plain_arrays(self) -> None:
        assert not has_jax_function((jnp.ones(3), None, 2))

    def test_has_jax_function_true_for_implementations(self) -> None:
        assert has_jax_function((jnp.ones(3), _Wrapper(jnp.ones(3))))

    def test_handle_jax_function_dispatches(self) -> None:
        wrapper = _Wrapper(jnp.arange(6.0))
        result = handle_jax_function(jax_sum, (wrapper,), wrapper)
        assert isinstance(result, _Wrapper)
        assert float(np.asarray(result.array)) == pytest.approx(15.0)

    def test_handle_jax_function_skips_not_implemented(self) -> None:
        class _NotImplementedWrapper(_Wrapper):
            @classmethod
            def __jax_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN206
                del cls, func, types, args, kwargs
                return NotImplemented

        first = _NotImplementedWrapper(jnp.arange(3.0))
        second = _Wrapper(jnp.arange(3.0))

        result = handle_jax_function(jax_sum, (first, second), second)
        assert isinstance(result, _Wrapper)

    def test_handle_jax_function_raises_when_unhandled(self) -> None:
        class _AlwaysNotImplemented(_Wrapper):
            @classmethod
            def __jax_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN206
                del cls, func, types, args, kwargs
                return NotImplemented

        wrapper = _AlwaysNotImplemented(jnp.arange(3.0))
        with pytest.raises(TypeError, match="no implementation found"):
            handle_jax_function(jax_sum, (wrapper,), wrapper)

    def test_handle_jax_function_raises_without_overloads(self) -> None:
        with pytest.raises(TypeError, match="no implementation found"):
            handle_jax_function(jax_sum, (jnp.ones(3),), jnp.ones(3))


class TestWrappers:
    """The ``jnp`` mirroring wrappers fall back to plain ``jax.numpy``."""

    def test_reshape_without_override(self) -> None:
        result = jax_reshape(jnp.arange(6), (2, 3))
        assert result.shape == (2, 3)

    def test_reduction_wrappers_without_override(self) -> None:
        array = jnp.arange(6.0)
        assert float(jax_sum(array)) == pytest.approx(15.0)
        assert float(jax_mean(array)) == pytest.approx(2.5)
        assert float(jax_std(array)) == pytest.approx(float(jnp.std(array)))
        assert float(jax_var(array)) == pytest.approx(float(jnp.var(array)))

    def test_average_with_weights(self) -> None:
        array = jnp.asarray([1.0, 2.0, 3.0])
        weights = jnp.asarray([1.0, 0.0, 1.0])
        assert float(jax_average(array, weights=weights)) == pytest.approx(2.0)

    def test_concatenate_and_stack_without_override(self) -> None:
        array = jnp.arange(3.0)
        assert jax_concatenate([array, array]).shape == (6,)
        assert jax_stack([array, array], 1).shape == (3, 2)

    def test_concatenate_dispatches_on_sequence_elements(self) -> None:
        wrapper = _Wrapper(jnp.arange(3.0))
        result = jax_concatenate([wrapper, wrapper])
        assert isinstance(result, _Wrapper)
        assert result.array.shape == (6,)


class TestImplementationDefaults:
    """The concrete methods that ``JaxLikeImplementation`` provides for free."""

    @staticmethod
    def _wrapper() -> _Wrapper:
        return _Wrapper(jnp.arange(6.0).reshape(2, 3))

    def test_transpose_property(self) -> None:
        assert self._wrapper().T.shape == (3, 2)

    def test_matrix_transpose_property(self) -> None:
        assert self._wrapper().mT.shape == (3, 2)

    def test_adjoint_property(self) -> None:
        result = self._wrapper().mH
        np.testing.assert_allclose(np.asarray(result), np.arange(6.0).reshape(2, 3).T)

    def test_reshape_with_separate_ints(self) -> None:
        assert self._wrapper().reshape(3, 2).shape == (3, 2)

    def test_reshape_with_sequence(self) -> None:
        assert self._wrapper().reshape((3, 2)).shape == (3, 2)

    def test_transpose_method_with_axes(self) -> None:
        assert self._wrapper().transpose(1, 0).shape == (3, 2)

    def test_transpose_method_without_axes(self) -> None:
        assert self._wrapper().transpose().shape == (3, 2)

    def test_ravel(self) -> None:
        assert self._wrapper().ravel().shape == (6,)

    def test_flatten(self) -> None:
        assert self._wrapper().flatten().shape == (6,)

    def test_squeeze(self) -> None:
        wrapper = _Wrapper(jnp.ones((1, 3)))
        assert wrapper.squeeze().shape == (3,)

    def test_squeeze_with_axis(self) -> None:
        wrapper = _Wrapper(jnp.ones((1, 3)))
        assert wrapper.squeeze(0).shape == (3,)

    def test_copy(self) -> None:
        wrapper = self._wrapper()
        copied = wrapper.copy()
        assert isinstance(copied, _Wrapper)
        np.testing.assert_allclose(np.asarray(copied), np.asarray(wrapper))

    def test_item(self) -> None:
        assert _Wrapper(jnp.asarray(3.0)).item() == pytest.approx(3.0)

    def test_tolist(self) -> None:
        assert _Wrapper(jnp.asarray([1.0, 2.0])).tolist() == [1.0, 2.0]

    def test_array_conversion(self) -> None:
        np.testing.assert_allclose(np.asarray(self._wrapper()), np.arange(6.0).reshape(2, 3))

    def test_array_conversion_with_dtype(self) -> None:
        assert np.asarray(self._wrapper(), dtype=np.int32).dtype == np.int32

    def test_astype(self) -> None:
        cast_wrapper = self._wrapper().astype(jnp.int32)
        assert isinstance(cast_wrapper, _Wrapper)
        assert cast_wrapper.array.dtype == jnp.int32

    def test_block_until_ready_returns_self(self) -> None:
        wrapper = self._wrapper()
        assert wrapper.block_until_ready() is wrapper

    def test_to_device(self) -> None:
        moved = self._wrapper().to_device(jax.devices()[0])
        assert isinstance(moved, _Wrapper)
        np.testing.assert_allclose(np.asarray(moved), np.arange(6.0).reshape(2, 3))

    def test_to_device_rejects_stream(self) -> None:
        with pytest.raises(NotImplementedError, match="stream"):
            self._wrapper().to_device(jax.devices()[0], stream=0)

    def test_stop_gradient_returns_same_values(self) -> None:
        detached = self._wrapper().stop_gradient()
        assert isinstance(detached, _Wrapper)
        np.testing.assert_allclose(np.asarray(detached), np.arange(6.0).reshape(2, 3))

    def test_stop_gradient_cuts_the_gradient(self) -> None:
        def loss(wrapper: _Wrapper) -> jax.Array:
            return jnp.sum(wrapper.stop_gradient().array) + jnp.sum(wrapper.array)

        gradient = jax.grad(loss)(self._wrapper())
        np.testing.assert_allclose(np.asarray(gradient.array), np.ones((2, 3)))

    def test_at_is_left_to_subclasses(self) -> None:
        """Mirrors ``TorchLikeImplementation`` leaving ``__setitem__`` to its subclasses."""
        assert not hasattr(JaxLikeImplementation, "at")


class TestPytreeRegistration:
    """Concrete subclasses are usable inside JAX transformations."""

    def test_tree_map_round_trip(self) -> None:
        wrapper = _Wrapper(jnp.arange(3.0))
        mapped = jax.tree.map(lambda leaf: leaf * 2, wrapper)
        assert isinstance(mapped, _Wrapper)
        np.testing.assert_allclose(np.asarray(mapped.array), [0.0, 2.0, 4.0])

    def test_static_fields_survive_flattening(self) -> None:
        wrapper = _Wrapper(jnp.arange(3.0), label="static")
        mapped = jax.tree.map(lambda leaf: leaf + 1, wrapper)
        assert mapped.label == "static"

    def test_jit_round_trip(self) -> None:
        @jax.jit
        def double(value: _Wrapper) -> _Wrapper:
            return _Wrapper(value.array * 2, value.label)

        result = double(_Wrapper(jnp.arange(3.0)))
        assert isinstance(result, _Wrapper)
        np.testing.assert_allclose(np.asarray(result.array), [0.0, 2.0, 4.0])

    def test_none_field_is_a_child(self) -> None:
        @dataclass(frozen=True)
        class _Optional(_Wrapper):
            weights: jax.Array | None = None

        wrapper = _Optional(jnp.arange(3.0))
        children, _ = wrapper.tree_flatten()
        assert children[-1] is None
