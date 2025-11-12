"""Tests for flax DropConnect models."""

from __future__ import annotations

import pytest

from probly.transformation import dropconnect
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402

from probly.transformation.dropconnect.flax import DropConnectDense


@pytest.fixture
def flax_linear_layer() -> nnx.Linear:
    return nnx.Linear(in_features=8, out_features=4, rngs=nnx.Rngs(params=0))


class TestNetworkArchitectures:
    def test_linear_network_replacement(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # count original layers
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        count_dropconnect_original = count_layers(flax_model_small_2d_2d, DropConnectDense)

        # count modified layers
        count_linear_modified = count_layers(model, nnx.Linear)
        count_dropconnect_modified = count_layers(model, DropConnectDense)

        # checks layer counts
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_dropconnect_original == 0
        assert count_linear_modified < count_linear_original
        assert count_linear_original == (count_linear_modified + count_dropconnect_modified)

    def test_convolutional_layers_untouched(self, flax_conv_linear_model: nnx.Sequential) -> None:
        p = 0.5
        model = dropconnect(flax_conv_linear_model, p)

        # count convoluntional layers
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv)
        count_conv_modified = count_layers(model, nnx.Conv)
        count_dropconnect_modified = count_layers(model, DropConnectDense)

        # check layer counts
        assert count_conv_original == count_conv_modified
        assert count_dropconnect_modified >= 1
        assert isinstance(model, type(flax_conv_linear_model))

    def test_custom_model(self, flax_custom_model: nnx.Module) -> None:
        p = 0.25
        model = dropconnect(flax_custom_model, p)
        assert isinstance(model, type(flax_custom_model))


class TestDropConnectParameters:
    def test_correct_p_value(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)
        for m in model.layers:
            if isinstance(m, DropConnectDense):
                assert pytest.approx(m.p, rel=1e-6) == p

    def test_weights_and_biases_preserved(self, flax_linear_layer: nnx.Linear) -> None:
        p = 0.5
        layer = DropConnectDense(flax_linear_layer, p=p, rngs=None)
        assert layer.weight.value.shape == flax_linear_layer.kernel.value.shape
        if layer.use_bias:
            assert layer.bias is not None
            assert layer.bias.value.shape == flax_linear_layer.bias.value.shape


class TestFunctionalBehavior:
    def test_forward_pass_shapes(self, flax_linear_layer: nnx.Linear) -> None:
        from flax.nnx import Rngs
        import jax
        import jax.numpy as jnp

        x = jnp.ones((3, flax_linear_layer.in_features))
        rngs = Rngs(dropout=jax.random.key(0))
        drop_layer = DropConnectDense(flax_linear_layer, p=0.3, rngs=rngs)
        y = drop_layer(x)
        assert y.shape == (3, flax_linear_layer.out_features)

    def test_training_vs_inference_behavior(self, flax_linear_layer: nnx.Linear) -> None:
        from flax.nnx import Rngs
        import jax
        import jax.numpy as jnp

        x = jnp.ones((5, flax_linear_layer.in_features))

        # Training mode (rngs provided)
        rngs = Rngs(dropout=jax.random.key(42))
        train_layer = DropConnectDense(flax_linear_layer, p=0.5, rngs=rngs)
        y1 = train_layer(x)
        y2 = train_layer(x)
        assert not jnp.allclose(y1, y2)

        # Inference mode (no rngs)
        eval_layer = DropConnectDense(flax_linear_layer, p=0.5, rngs=None)
        y1_eval = eval_layer(x)
        y2_eval = eval_layer(x)
        assert jnp.allclose(y1_eval, y2_eval)
