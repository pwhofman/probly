"""Tests for flax batchensemble models."""

from __future__ import annotations

from typing import cast

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.layers.flax import BatchEnsembleConv, BatchEnsembleLinear  # noqa: E402
from probly.method.batchensemble import batchensemble  # noqa: E402
from probly.predictor import predict  # noqa: E402
from probly.representation.sample.jax import JaxArraySample  # noqa: E402
from tests.probly.flax_utils import count_layers  # noqa: E402


class TestBatchEnsembleLayerAttributes:
    """Tests for attributes of BatchEnsembleLinear and BatchEnsembleConv layers."""

    def test_batchensemble_linear_attributes(self) -> None:
        """Tests BatchEnsembleLinear layer attributes."""
        rngs = nnx.Rngs(0, params=1)
        linear_layer = nnx.Linear(3, 4, rngs=rngs)

        num_members = 2
        use_base_weights = False
        r_mean = 0.99
        r_std = 0.02
        s_mean = 0.99
        s_std = 0.02
        # ``batchensemble`` traverses the input model in place and tags it with num_members;
        # for a single ``nnx.Linear`` input the result is a ``BatchEnsembleLinear`` directly.
        batchensemble_linear = cast(
            "BatchEnsembleLinear",
            batchensemble(
                base=linear_layer,
                num_members=num_members,
                use_base_weights=use_base_weights,
                init="normal",
                r_mean=r_mean,
                r_std=r_std,
                s_mean=s_mean,
                s_std=s_std,
                rngs=rngs,
            ),
        )

        assert not jnp.equal(
            batchensemble_linear.kernel.value, linear_layer.kernel.value
        ).all()  # use_base_weights = False
        # Per-member bias initialized by broadcasting the base layer's bias across members.
        assert batchensemble_linear.bias.shape == (num_members, linear_layer.out_features)
        assert linear_layer.bias is not None
        expected_bias = jnp.broadcast_to(
            linear_layer.bias.value[None, :],
            (num_members, linear_layer.out_features),
        )
        assert jnp.equal(batchensemble_linear.bias.value, expected_bias).all()
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
        # r modulates input dim, s modulates output dim (paper convention).
        assert batchensemble_linear.r.shape == (num_members, batchensemble_linear.in_features)
        assert batchensemble_linear.s.shape == (num_members, batchensemble_linear.out_features)

        # Simulate the rng draws to verify init is the Gaussian formula expected.
        new_rngs = nnx.Rngs(0, params=1)
        _ = new_rngs.params()  # linear kernel key
        _ = new_rngs.params()  # linear bias key
        kernel_key = new_rngs.params()
        kernel_init = jax.nn.initializers.lecun_normal()
        kernel = kernel_init(
            kernel_key, (linear_layer.in_features, linear_layer.out_features), linear_layer.param_dtype
        )
        assert jnp.equal(batchensemble_linear.kernel.value, kernel).all()

        # r is drawn before s in the layer __init__.
        r_key = new_rngs.params()
        expected_r = r_mean + r_std * jax.random.normal(
            r_key,
            (num_members, batchensemble_linear.in_features),
            dtype=linear_layer.param_dtype,
        )
        assert jnp.equal(batchensemble_linear.r.value, expected_r).all()

        s_key = new_rngs.params()
        expected_s = s_mean + s_std * jax.random.normal(
            s_key,
            (num_members, batchensemble_linear.out_features),
            dtype=linear_layer.param_dtype,
        )
        assert jnp.equal(batchensemble_linear.s.value, expected_s).all()

    def test_batchensemble_conv_attributes(self) -> None:
        """Tests BatchEnsembleConv layer attributes."""
        rngs = nnx.Rngs(0, params=1)
        conv_layer = nnx.Conv(3, 4, (1, 1), rngs=rngs)

        num_members = 2
        use_base_weights = False
        r_mean = 0.99
        r_std = 0.02
        s_mean = 0.99
        s_std = 0.02
        batchensemble_conv = cast(
            "BatchEnsembleConv",
            batchensemble(
                base=conv_layer,
                num_members=num_members,
                use_base_weights=use_base_weights,
                init="normal",
                r_mean=r_mean,
                r_std=r_std,
                s_mean=s_mean,
                s_std=s_std,
                rngs=rngs,
            ),
        )

        assert batchensemble_conv.kernel_shape == conv_layer.kernel_shape
        assert not jnp.equal(batchensemble_conv.kernel.value, conv_layer.kernel.value).all()  # use_base_weights = False
        # Per-member bias initialized by broadcasting the base layer's bias across members.
        assert batchensemble_conv.bias.shape == (num_members, conv_layer.out_features)
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
        # r modulates input channels, s modulates output channels (paper convention).
        assert batchensemble_conv.r.shape == (num_members, batchensemble_conv.in_features)
        assert batchensemble_conv.s.shape == (num_members, batchensemble_conv.out_features)

        # Simulate the rng draws to verify init is the Gaussian formula expected.
        new_rngs = nnx.Rngs(0, params=1)
        _ = new_rngs.params()  # conv kernel key
        _ = new_rngs.params()  # conv bias key
        kernel_key = new_rngs.params()
        kernel_init = jax.nn.initializers.lecun_normal()
        kernel = kernel_init(kernel_key, batchensemble_conv.kernel_shape, batchensemble_conv.param_dtype)
        assert jnp.equal(batchensemble_conv.kernel.value, kernel).all()

        # r is drawn before s in the layer __init__.
        r_key = new_rngs.params()
        expected_r = r_mean + r_std * jax.random.normal(
            r_key,
            (num_members, batchensemble_conv.in_features),
            dtype=conv_layer.param_dtype,
        )
        assert jnp.equal(batchensemble_conv.r.value, expected_r).all()

        s_key = new_rngs.params()
        expected_s = s_mean + s_std * jax.random.normal(
            s_key,
            (num_members, batchensemble_conv.out_features),
            dtype=conv_layer.param_dtype,
        )
        assert jnp.equal(batchensemble_conv.s.value, expected_s).all()

    def test_batchensemble_bias_none(self) -> None:
        rngs = nnx.Rngs(0, params=1)
        linear = nnx.Linear(1, 2, rngs=rngs, use_bias=False)
        conv = nnx.Conv(1, 2, (1), rngs=rngs, use_bias=False)

        batchensemble_linear = batchensemble(linear)
        batchensemble_conv = batchensemble(conv)

        # Single layer in -> single BatchEnsembleLinear / BatchEnsembleConv out.
        assert isinstance(batchensemble_linear, BatchEnsembleLinear)
        assert isinstance(batchensemble_conv, BatchEnsembleConv)
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
        # The traversed model preserves the input's class and is tagged with num_members.
        assert isinstance(batchensemble_model, type(model))
        assert batchensemble_model.num_members == num_members

        count_linear_original = count_layers(model, nnx.Linear)
        count_conv_original = count_layers(model, nnx.Conv)
        count_sequential_original = count_layers(model, nnx.Sequential)

        count_batchensemblelinear_modified = count_layers(batchensemble_model, BatchEnsembleLinear)
        count_batchensembleconv_modified = count_layers(batchensemble_model, BatchEnsembleConv)
        count_sequential_modified = count_layers(batchensemble_model, nnx.Sequential)

        assert count_sequential_modified == count_sequential_original
        assert count_batchensemblelinear_modified == count_linear_original
        assert count_batchensembleconv_modified == count_conv_original

    def test_custom_model(self, flax_custom_model) -> None:
        num_members = 5
        model = batchensemble(flax_custom_model, num_members)
        # Type-preserving: the result is the same class as the input.
        assert isinstance(model, type(flax_custom_model))
        assert not isinstance(model, nnx.Sequential)
        assert model.num_members == num_members

        count_linear_original = count_layers(flax_custom_model, nnx.Linear)
        count_conv_original = count_layers(flax_custom_model, nnx.Conv)
        count_sequential_original = count_layers(flax_custom_model, nnx.Sequential)

        count_batchensemblelinear_modified = count_layers(model, BatchEnsembleLinear)
        count_batchensembleconv_modified = count_layers(model, BatchEnsembleConv)
        count_sequential_modified = count_layers(model, nnx.Sequential)

        assert count_batchensemblelinear_modified == count_linear_original
        assert count_batchensembleconv_modified == count_conv_original
        assert count_sequential_modified == count_sequential_original


class TestBatchEnsembleCalls:
    """Test class for BatchEnsemble layer calls."""

    def test_batchensemble_layer_calls(self) -> None:
        """Tests direct calls to BatchEnsembleLinear and BatchEnsembleConv layers.

        BatchEnsemble layers expect ``[E*B, ...]`` inputs and return ``[E*B, ...]``
        outputs (the "pure" forward signature). The user-facing ``predict()`` handler
        wraps these into a :class:`JaxArraySample` with ``sample_axis=0``.
        """
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

        # Direct call: pre-tile by num_members.
        be_linear_out = batchensemble_linear(jnp.tile(x_linear, (num_members, 1)))
        be_conv1d_out = batchensemble_conv1d(jnp.tile(x_conv1d, (num_members, 1, 1)))
        be_conv2d_out = batchensemble_conv2d(jnp.tile(x_conv2d, (num_members, 1, 1, 1)))
        be_conv3d_out = batchensemble_conv3d(jnp.tile(x_conv3d, (num_members, 1, 1, 1, 1)))
        assert be_linear_out.shape == (num_members * batch_size, out_dim)
        assert be_conv1d_out.shape == (num_members * batch_size, 1, out_dim)
        assert be_conv2d_out.shape == (num_members * batch_size, 1, 1, out_dim)
        assert be_conv3d_out.shape == (num_members * batch_size, 1, 1, 1, out_dim)

        # predict() wraps in a JaxArraySample with sample_axis=0 and shape [E, B, ...].
        linear_sample = predict(batchensemble_linear, x_linear)
        assert isinstance(linear_sample, JaxArraySample)
        assert linear_sample.sample_axis == 0
        assert linear_sample.array.shape == (num_members, batch_size, out_dim)

    def test_batchensemble_flax_custom_model_call(self, flax_custom_model) -> None:
        """Tests call of transformed flax_custom_model via predict()."""
        rngs = nnx.Rngs(0, params=1)
        num_members = 5
        batchensemble_model = batchensemble(
            flax_custom_model, num_members=num_members, use_base_weights=True, rngs=rngs
        )

        # The model now expects ``[E*B, ...]``; use predict() so the tile/un-tile and
        # Sample-wrap are handled for you.
        x = jnp.ones((1, 10))
        sample = predict(batchensemble_model, x)
        assert isinstance(sample, JaxArraySample)
        assert sample.sample_axis == 0
        assert sample.array.shape == (num_members, 1, 4)

    def test_batchensemble_call_errors(self) -> None:
        """Wrong-rank inputs to the model are rejected by the BatchEnsemble layer."""
        rngs = nnx.Rngs(0, params=1)
        linear = nnx.Linear(1, 2, rngs=rngs)
        conv = nnx.Conv(1, 2, 1, rngs=rngs)

        num_members = 2
        batchensemble_linear = batchensemble(linear, num_members=num_members)
        batchensemble_conv = batchensemble(conv, num_members=num_members)

        # 3D input to BatchEnsembleLinear.
        with pytest.raises(ValueError, match=r"Expected 2D input \[E\*B, in_features\]"):
            batchensemble_linear(jnp.ones((2, 1, 1)))

        # 4D input to a 1D-conv BatchEnsembleConv.
        with pytest.raises(ValueError, match=r"Expected 3D input"):
            batchensemble_conv(jnp.ones((2, 1, 1, 1)))

        # Batch size not divisible by num_members.
        with pytest.raises(ValueError, match=r"is not divisible by num_members"):
            batchensemble_linear(jnp.ones((3, 1)))
