"""Test for flax subensemble models."""

from __future__ import annotations

import pytest

from probly.transformation.subensemble import subensemble
from tests.probly.flax_utils import count_layers

jax = pytest.importorskip("jax")
from jax import numpy as jnp  # noqa: E402

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


class TestGeneration:
    """Tests for different subensemble generations."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "flax_model_small_2d_2d",
            "flax_conv_linear_model",
            "flax_regression_model_1d",
            "flax_regression_model_2d",
            "flax_dropout_model",
        ],
    )
    def test_subensemble_default(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Test for default subensemble generation."""
        model = request.getfixturevalue(model_fixture)
        num_heads = 5
        head_layer = 1  # default

        subensemble_model = subensemble(model, num_heads=num_heads)
        backbone = subensemble_model[0]
        heads = subensemble_model[1]

        count_linear_original = count_layers(model, nnx.Linear)
        count_linear_backbone = count_layers(backbone, nnx.Linear)
        count_linear_heads = count_layers(heads, nnx.Linear)
        count_sequential_original = count_layers(model, nnx.Sequential)
        count_sequential_backbone = count_layers(backbone, nnx.Sequential)
        count_sequential_heads = count_layers(heads, nnx.Sequential)
        count_convolution_original = count_layers(model, nnx.Conv)
        count_convolution_backbone = count_layers(backbone, nnx.Conv)
        count_convolution_heads = count_layers(heads, nnx.Conv)

        assert isinstance(subensemble_model, nnx.Module)
        assert isinstance(backbone, nnx.Sequential)
        assert isinstance(heads, nnx.List)
        assert count_linear_heads == num_heads
        assert count_linear_backbone == (count_linear_original - head_layer)
        assert (count_linear_original + num_heads - 1) == count_linear_heads + count_linear_backbone
        assert count_sequential_heads == num_heads
        assert count_sequential_backbone == count_sequential_original
        assert count_convolution_backbone == count_convolution_original
        assert count_convolution_heads == 0

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "flax_model_small_2d_2d",
            "flax_conv_linear_model",
            "flax_regression_model_1d",
            "flax_regression_model_2d",
        ],
    )
    def test_subensemble_2_head_layers(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Test for 2 head layers subensemble generation."""
        model = request.getfixturevalue(model_fixture)
        num_heads = 5
        head_layer = 2

        subensemble_model = subensemble(
            model,
            num_heads=num_heads,
            head_layer=head_layer,
        )
        backbone = subensemble_model[0]
        heads = subensemble_model[1]

        count_linear_original = count_layers(model, nnx.Linear)
        count_linear_backbone = count_layers(backbone, nnx.Linear)
        count_linear_heads = count_layers(heads, nnx.Linear)
        count_sequential_original = count_layers(model, nnx.Sequential)
        count_sequential_backbone = count_layers(backbone, nnx.Sequential)
        count_sequential_heads = count_layers(heads, nnx.Sequential)
        count_convolution_original = count_layers(model, nnx.Conv)
        count_convolution_backbone = count_layers(backbone, nnx.Conv)
        count_convolution_heads = count_layers(heads, nnx.Conv)

        assert isinstance(subensemble_model, nnx.Module)
        assert isinstance(backbone, nnx.Sequential)
        assert isinstance(heads, nnx.List)
        if isinstance(model.layers[-2], nnx.Linear):
            assert count_linear_heads == (num_heads * head_layer)
        else:
            assert count_sequential_heads == num_heads
        if isinstance(model.layers[-2], nnx.Linear):
            assert count_linear_backbone == (count_linear_original - head_layer)
        else:
            assert count_linear_backbone == (count_linear_original - 1)
        if isinstance(model.layers[-2], nnx.Linear):
            assert count_linear_original == (count_linear_heads - (num_heads * head_layer) + 2) + count_linear_backbone
        else:
            assert count_linear_original == (count_linear_heads - num_heads + 1) + count_linear_backbone
        assert count_sequential_heads == num_heads
        assert count_sequential_backbone == count_sequential_original
        assert count_convolution_backbone == count_convolution_original
        assert count_convolution_heads == 0

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "flax_model_small_2d_2d",
            "flax_conv_linear_model",
            "flax_regression_model_1d",
            "flax_regression_model_2d",
        ],
    )
    def test_subensemble_with_head_model(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Test for backbone and head model subensemble generation."""
        model = request.getfixturevalue(model_fixture)
        num_heads = 5

        subensemble_model = subensemble(
            base=model,
            num_heads=num_heads,
            head=model,
        )
        backbone = subensemble_model[0]
        heads = subensemble_model[1]

        count_linear_original = count_layers(model, nnx.Linear)
        count_linear_backbone = count_layers(backbone, nnx.Linear)
        count_linear_heads = count_layers(heads, nnx.Linear)
        count_sequential_original = count_layers(model, nnx.Sequential)
        count_sequential_backbone = count_layers(backbone, nnx.Sequential)
        count_sequential_heads = count_layers(heads, nnx.Sequential)
        count_convolution_original = count_layers(model, nnx.Conv)
        count_convolution_backbone = count_layers(backbone, nnx.Conv)
        count_convolution_heads = count_layers(heads, nnx.Conv)

        assert isinstance(subensemble_model, nnx.Module)
        assert isinstance(backbone, nnx.Sequential)
        assert isinstance(heads, nnx.List)
        assert count_linear_heads == (count_linear_original * num_heads)
        assert count_linear_backbone == count_linear_original
        assert (
            (count_linear_original * num_heads) + count_linear_original
        ) == count_linear_heads + count_linear_backbone
        assert count_sequential_heads == num_heads
        assert count_sequential_backbone == count_sequential_original
        assert count_convolution_backbone == count_convolution_original
        assert count_convolution_heads == (count_convolution_original * num_heads)


class TestReset:
    """Test class for parameter resetting."""

    @pytest.mark.skip(reason="Not yet implemented, waiting for flax ensemble merge.")
    def test_parameters_reset(
        self,
        flax_model_small_2d_2d: nnx.Sequential,
    ) -> None:
        """Tests if the parameters of the model are reset correctly."""
        num_heads = 2
        model = flax_model_small_2d_2d

        original_params_last_layer = jax.tree_util.tree_leaves(model.layers[-1])

        subensemble_result = subensemble(model, num_heads=num_heads, reset_params=True)
        heads = subensemble_result[1]

        first_head = heads[0]
        params_first_head = jax.tree_util.tree_leaves(first_head)

        for head in heads:
            params_head = jax.tree_util.tree_leaves(head)

            for original_params_last_layer_iterator, params_first_head_iterator in zip(
                original_params_last_layer[1],
                params_head[1],
                strict=False,
            ):
                assert not jnp.array_equal(original_params_last_layer_iterator, params_first_head_iterator)

            for params_first_head_iterator, params_head_iterator in zip(
                params_first_head[1],
                params_head[1],
                strict=False,
            ):
                if head is first_head:
                    continue
                assert not jnp.array_equal(params_first_head_iterator, params_head_iterator)

    def test_parameters_not_reset(
        self,
        flax_model_small_2d_2d: nnx.Sequential,
    ) -> None:
        """Tests if the parameters of the model are not reset when specified."""
        num_heads = 2
        model = flax_model_small_2d_2d

        original_params_last_layer = jax.tree_util.tree_leaves(model.layers[-1])

        subensemble_result = subensemble(
            model,
            num_heads=num_heads,
            reset_params=False,
        )
        heads = subensemble_result[1]

        for head in heads:
            params_head = jax.tree_util.tree_leaves(head)
            for original_params_last_layer_iterator, params_head_iterator in zip(
                original_params_last_layer[1],
                params_head[1],
                strict=False,
            ):
                assert jnp.array_equal(original_params_last_layer_iterator, params_head_iterator)

    def test_forward_pass_shape(
        self,
        flax_model_small_2d_2d: nnx.Sequential,
    ) -> None:
        """Tests if the forward pass works correctly after subensemble transformation."""
        num_heads = 2

        subensemble_result = subensemble(
            flax_model_small_2d_2d,
            num_heads=num_heads,
            reset_params=True,
        )
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        x = jnp.ones((1, 2))

        backbone_output = backbone(x) if backbone is not None else x

        # forward pass through each head
        for head in heads:
            head_output = head(backbone_output)
            expected_output_shape = (1, head.layers[-1].out_features)
            assert head_output.shape == expected_output_shape


class TestEdgeCases:
    """Tests for edge-case configurations."""

    def test_zero_heads(
        self,
        flax_model_small_2d_2d: nnx.Module,
    ) -> None:
        """num_heads = 0 should split the model and return an empty list of heads."""
        num_heads = 0
        head_layer = 1

        subensemble_result = subensemble(flax_model_small_2d_2d, num_heads=num_heads)
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        original_layers = count_layers(flax_model_small_2d_2d, nnx.Module)
        backbone_layers = count_layers(backbone, nnx.Module)

        assert isinstance(subensemble_result, nnx.Module)
        assert isinstance(backbone, nnx.Sequential)
        assert isinstance(heads, nnx.List)
        assert len(heads) == num_heads
        assert backbone_layers == original_layers - head_layer
        for head in heads:
            head_layers = count_layers(head, nnx.Module)
            assert head_layers == 0

    def test_invalid_head_layer(
        self,
        flax_model_small_2d_2d: nnx.Module,
    ) -> None:
        """Test if head_layer <= 0 raises ValueError."""
        num_heads = 4
        head_layer = 0

        with pytest.raises(ValueError, match="head_layer must be a positive number, but got head_layer=0 instead"):
            subensemble(
                flax_model_small_2d_2d,
                num_heads=num_heads,
                head_layer=head_layer,
            )

    def test_large_head_layer(
        self,
        flax_model_small_2d_2d: nnx.Module,
    ) -> None:
        """Test if backbone can be empty while head is an ensemble of the base model."""
        num_heads = 4
        head_layer = count_layers(flax_model_small_2d_2d, nnx.Module)

        subensemble_result = subensemble(
            flax_model_small_2d_2d,
            num_heads=num_heads,
            head_layer=head_layer,
        )
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        original_layers = count_layers(flax_model_small_2d_2d, nnx.Module)
        backbone_layers = count_layers(backbone, nnx.Module)

        assert isinstance(subensemble_result, nnx.Module)
        assert isinstance(backbone, nnx.Sequential)
        assert isinstance(heads, nnx.List)
        assert len(heads) == num_heads
        assert backbone_layers == 2  # Sequential + List
        for head in heads:
            head_layers = count_layers(head, nnx.Module)
            assert head_layers == original_layers
