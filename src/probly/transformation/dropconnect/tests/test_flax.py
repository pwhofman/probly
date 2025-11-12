import pytest
jax = pytest.importorskip("jax")
import jax.numpy as jnp
from probly.transformation.dropconnect.flax import DropConnectDense, replace_flax_dropconnect


class DummyParam:
    def __init__(self, value):
        self.value = value


class DummyLinear:
    def __init__(self, in_features, out_features, use_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.kernel = DummyParam(jnp.ones((in_features, out_features), dtype=jnp.float32))
        self.bias = DummyParam(jnp.zeros((out_features,), dtype=jnp.float32))


def test_init_copies_attributes():
    base = DummyLinear(3, 2)
    m = DropConnectDense(base_layer=base, p=0.2)
    assert m.in_features == 3
    assert m.out_features == 2
    assert jnp.array_equal(m.weight.value, base.kernel.value)


def test_output_shape_is_correct():
    base = DummyLinear(4, 3)
    m = DropConnectDense(base_layer=base, p=0.0)
    x = jnp.ones((2, 4), dtype=jnp.float32)
    y = m(x)
    assert y.shape == (2, 3)


def test_inference_scales_weights_correctly():
    base = DummyLinear(2, 2, use_bias=False)
    m = DropConnectDense(base_layer=base, p=0.5)
    x = jnp.ones((1, 2), dtype=jnp.float32)
    y = m(x)
    expected = jnp.dot(x, base.kernel.value * 0.5)
    assert jnp.allclose(y, expected)


def test_replace_function_returns_model():
    base = DummyLinear(2, 2)
    m = replace_flax_dropconnect(base, p=0.1)
    assert isinstance(m, DropConnectDense)
    assert m.p == 0.1
