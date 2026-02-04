"""Tests for flax batchensemble models."""

from __future__ import annotations

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.layers.flax import BatchEnsembleConv, BatchEnsembleLinear  # noqa: E402
from probly.transformation.batchensemble import batchensemble  # noqa: E402
from tests.probly.flax_utils import count_layers  # noqa: E402


class TestBatchEnsembleLayerAttributes:
    """Tests for attributes of BatchEnsembleLinear and BatchEnsembleConv layers."""

    def test_batchensemble_linear_attributes(self) -> None:
        """Tests BatchEnsembleLinear layer attributes."""
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(2, 2, rngs=rngs)

        num_members = 2
        use_base_weights = False
        s_mean = 0.99
        s_std = 0.02
        r_mean = 0.99
        r_std = 0.02
        batchensemble_linear = batchensemble(
            base=linear_layer,
            num_members=num_members,
            use_base_weights=use_base_weights,
            s_mean=s_mean,
            s_std=s_std,
            r_mean=r_mean,
            r_std=r_std,
            rngs=rngs,
        )

        assert not jnp.equal(
            batchensemble_linear.kernel.value, linear_layer.kernel.value
        ).all()  # use_base_weights = False
        assert jnp.equal(batchensemble_linear.bias, linear_layer.bias).all()
        assert batchensemble_linear.in_features == linear_layer.in_features
        assert batchensemble_linear.out_features == linear_layer.out_features
        assert batchensemble_linear.use_bias == linear_layer.use_bias
        assert batchensemble_linear.dtype == linear_layer.dtype
        assert batchensemble_linear.param_dtype == linear_layer.param_dtype
        assert batchensemble_linear.precision == linear_layer.precision
        assert batchensemble_linear.dot_general == linear_layer.dot_general
        assert batchensemble_linear.promote_dtype == linear_layer.promote_dtype
        assert batchensemble_linear.preferred_element_type == linear_layer.preferred_element_type

        assert batchensemble_linear.num_members == num_members
        assert batchensemble_linear.r.shape == (num_members, batchensemble_linear.in_features)
        assert batchensemble_linear.s.shape == (num_members, batchensemble_linear.out_features)

        # simulate rngs
        new_rngs = nnx.Rngs(0, params=1)
        _ = new_rngs.params()  # linear kernel key
        _ = new_rngs.params()  # linear bias key
        kernel_key = new_rngs.params()
        kernel_init = jax.nn.initializers.lecun_normal()
        kernel = kernel_init(
            kernel_key, (linear_layer.in_features, linear_layer.out_features), linear_layer.param_dtype
        )
        assert jnp.equal(batchensemble_linear.kernel.value, kernel).all()

        s_key = new_rngs.params()
        assert jnp.equal(
            batchensemble_linear.s.value,
            s_mean + s_std * jax.random.normal(s_key, (num_members, batchensemble_linear.in_features)),
        ).all()

        r_key = new_rngs.params()
        assert jnp.equal(
            batchensemble_linear.r.value,
            r_mean + r_std * jax.random.normal(r_key, (num_members, batchensemble_linear.in_features)),
        ).all()

    def test_batchensemble_conv_attributes(self) -> None:
        """Tests BatchEnsembleConv layer attributes."""
        rngs = nnx.Rngs(0, params=1)
        conv_layer = nnx.Conv(2, 2, (1, 1), rngs=rngs)

        num_members = 2
        use_base_weights = False
        s_mean = 0.99
        s_std = 0.02
        r_mean = 0.99
        r_std = 0.02
        batchensemble_conv = batchensemble(
            base=conv_layer,
            num_members=num_members,
            use_base_weights=use_base_weights,
            s_mean=s_mean,
            s_std=s_std,
            r_mean=r_mean,
            r_std=r_std,
            rngs=rngs,
        )

        assert batchensemble_conv.kernel_shape == conv_layer.kernel_shape
        assert not jnp.equal(batchensemble_conv.kernel.value, conv_layer.kernel.value).all()  # use_base_weights = False
        assert jnp.equal(batchensemble_conv.bias, conv_layer.bias).all()
        assert batchensemble_conv.in_features == conv_layer.in_features
        assert batchensemble_conv.out_features == conv_layer.out_features
        assert batchensemble_conv.kernel_size == conv_layer.kernel_size
        assert batchensemble_conv.strides == conv_layer.strides
        assert batchensemble_conv.padding == conv_layer.padding
        assert batchensemble_conv.input_dilation == conv_layer.input_dilation
        assert batchensemble_conv.kernel_dilation == conv_layer.kernel_dilation
        assert batchensemble_conv.feature_group_count == conv_layer.feature_group_count
        assert batchensemble_conv.use_bias == conv_layer.use_bias
        assert batchensemble_conv.mask == conv_layer.mask
        assert batchensemble_conv.dtype == conv_layer.dtype
        assert batchensemble_conv.param_dtype == conv_layer.param_dtype
        assert batchensemble_conv.precision == conv_layer.precision
        assert batchensemble_conv.conv_general_dilated == conv_layer.conv_general_dilated
        assert batchensemble_conv.promote_dtype == conv_layer.promote_dtype
        assert batchensemble_conv.preferred_element_type == conv_layer.preferred_element_type

        assert batchensemble_conv.num_members == num_members
        assert batchensemble_conv.s.shape == (num_members, batchensemble_conv.in_features)
        assert batchensemble_conv.r.shape == (num_members, batchensemble_conv.in_features)

        # simulate rngs
        new_rngs = nnx.Rngs(0, params=1)
        _ = new_rngs.params()  # conv kernel key
        _ = new_rngs.params()  # conv bias key
        kernel_key = new_rngs.params()
        kernel_init = jax.nn.initializers.lecun_normal()
        kernel = kernel_init(kernel_key, batchensemble_conv.kernel_shape, batchensemble_conv.param_dtype)
        assert jnp.equal(batchensemble_conv.kernel.value, kernel).all()

        s_key = new_rngs.params()
        assert jnp.equal(
            batchensemble_conv.s.value,
            s_mean + s_std * jax.random.normal(s_key, (num_members, batchensemble_conv.in_features)),
        ).all()
        r_key = new_rngs.params()
        assert jnp.equal(
            batchensemble_conv.r.value,
            r_mean + r_std * jax.random.normal(r_key, (num_members, batchensemble_conv.in_features)),
        ).all()

    def test_batchensemble_bias_none(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear = nnx.Linear(1, 2, rngs=rngs, use_bias=False)
        conv = nnx.Conv(1, 2, (1), rngs=rngs, use_bias=False)

        batchensemble_linear = batchensemble(linear)
        batchensemble_conv = batchensemble(conv)

        assert batchensemble_linear.bias is None
        assert batchensemble_conv.bias is None


class TestBatchEnsembleTransformation:
    """Test class for BatchEnsemble transformation."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "flax_model_small_2d_2d",
            "flax_conv1d_linear_model",
            "flax_conv_linear_model",
            "flax_conv3d_linear_model",
            "flax_regression_model_1d",
            "flax_regression_model_2d",
        ],
    )
    def test_fixtures(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        model = request.getfixturevalue(model_fixture)
        num_members = 5

        batchensemble_model = batchensemble(model, num_members)

        count_linear_original = count_layers(model, nnx.Linear)
        count_conv_original = count_layers(model, nnx.Conv)
        count_sequential_original = count_layers(model, nnx.Sequential)

        count_batchensemblelinear_modified = count_layers(batchensemble_model, BatchEnsembleLinear)
        count_batchensembleconv_modified = count_layers(batchensemble_model, BatchEnsembleConv)
        count_sequential_modified = count_layers(batchensemble_model, nnx.Sequential)

        assert isinstance(batchensemble_model, type(model))
        assert count_sequential_modified == count_sequential_original
        assert count_batchensemblelinear_modified == count_linear_original
        assert count_batchensembleconv_modified == count_conv_original

    def test_custom_model(self, flax_custom_model) -> None:
        num_members = 5
        model = batchensemble(flax_custom_model, num_members)

        count_linear_original = count_layers(flax_custom_model, nnx.Linear)
        count_conv_original = count_layers(flax_custom_model, nnx.Conv)
        count_sequential_original = count_layers(flax_custom_model, nnx.Sequential)

        count_batchensemblelinear_modified = count_layers(model, BatchEnsembleLinear)
        count_batchensembleconv_modified = count_layers(model, BatchEnsembleConv)
        count_sequential_modified = count_layers(model, nnx.Sequential)

        assert isinstance(model, type(flax_custom_model))
        assert not isinstance(model, nnx.Sequential)
        assert count_batchensemblelinear_modified == count_linear_original
        assert count_batchensembleconv_modified == count_conv_original
        assert count_sequential_modified == count_sequential_original


class TestBatchEnsembleCalls:
    """Test class for BatchEnsemble layer calls."""

    def test_batchensemble_layer_calls(self) -> None:
        """Tests calls of BatchEnsembleLinear and BatchEnsembleConv with 1D, 2D and 3D Convolutions."""
        rngs = nnx.Rngs(0, params=1)
        batch_size = 1
        out_dim = 2
        x_linear = jnp.ones((batch_size, out_dim))
        x_conv1d = jnp.ones((batch_size, 1, out_dim))
        x_conv2d = jnp.ones((batch_size, 1, 1, out_dim))
        x_conv3d = jnp.ones((batch_size, 1, 1, 1, out_dim))

        num_members = 5
        batchensemble_linear = batchensemble(
            nnx.Linear(2, 2, rngs=rngs), num_members=num_members, use_base_weights=True, rngs=rngs
        )
        batchensemble_conv1d = batchensemble(
            nnx.Conv(2, 2, 2, rngs=rngs), num_members=num_members, use_base_weights=True, rngs=rngs
        )
        batchensemble_conv2d = batchensemble(
            nnx.Conv(2, 2, (2, 2), rngs=rngs), num_members=num_members, use_base_weights=True, rngs=rngs
        )
        batchensemble_conv3d = batchensemble(
            nnx.Conv(2, 2, (2, 2, 2), rngs=rngs), num_members=num_members, use_base_weights=True, rngs=rngs
        )

        be_linear_out = batchensemble_linear(x_linear)
        be_conv1d_out = batchensemble_conv1d(x_conv1d)
        be_conv2d_out = batchensemble_conv2d(x_conv2d)
        be_conv3d_out = batchensemble_conv3d(x_conv3d)

        # check out not None
        assert be_linear_out is not None
        assert be_conv1d_out is not None
        assert be_conv2d_out is not None
        assert be_conv3d_out is not None

        # check out shapes
        assert be_linear_out.shape == (num_members, batch_size, out_dim)  # num_members, batch_size, out_dim
        assert be_conv1d_out.shape == (
            num_members,
            batch_size,
            1,
            out_dim,
        )  # num_members, batch_size, kernel_size(x), out_dim
        assert be_conv2d_out.shape == (
            num_members,
            batch_size,
            1,
            1,
            out_dim,
        )  # num_members, batch_size, kernel_size(x, x), out_dim
        assert be_conv3d_out.shape == (
            num_members,
            batch_size,
            1,
            1,
            1,
            out_dim,
        )  # num_members, batch_size , kernel_size(x, x, x), out_dim

    def test_batchensemble_flax_custom_model_call(self, flax_custom_model) -> None:
        """Tests call of transformed flax_custom_model."""
        rngs = nnx.Rngs(0, params=1)
        num_members = 5
        batchensemble_model = batchensemble(
            flax_custom_model, num_members=num_members, use_base_weights=True, rngs=rngs
        )

        x = jnp.ones((10,))
        out = batchensemble_model(x)

        assert out is not None
        assert out.shape == (num_members, 1, 4)  # num_members, batch_size , out_dim

    def test_batchensemble_call_errors(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear = nnx.Linear(1, 2, rngs=rngs)
        conv = nnx.Conv(1, 2, 1, rngs=rngs)

        num_members = 1
        batchensemble_linear = batchensemble(linear, num_members=num_members)
        batchensemble_conv = batchensemble(conv, num_members=num_members)

        x_linear = jnp.ones((2, 1, 1))
        x_conv = jnp.ones((2, 1, 1, 1))

        msg_linear = f"Expected first dim={num_members}, got {x_linear.shape[0]}"
        with pytest.raises(ValueError, match=msg_linear):
            batchensemble_linear(x_linear)

        msg_conv = f"Expected first dim={num_members}, got {x_conv.shape[0]}"
        with pytest.raises(ValueError, match=msg_conv):
            batchensemble_conv(x_conv)
