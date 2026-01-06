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
        ],
    )
    def test_subensemble_default(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Test for default subensemble generation."""
        model = request.getfixturevalue(model_fixture)
        num_heads = 2

        subensemble_model = subensemble(model, num_heads=num_heads)

        count_linear_original = count_layers(model, nnx.Linear)
        count_convolution_original = count_layers(model, nnx.Conv)

        assert isinstance(subensemble_model, nnx.Module)
        assert len(subensemble_model) == num_heads

        for member in subensemble_model:
            count_linear_subensemble = count_layers(member, nnx.Linear)
            count_convolutional_subensemble = count_layers(member, nnx.Conv)
            assert count_linear_subensemble == count_linear_original
            assert count_convolutional_subensemble == count_convolution_original

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
        num_heads = 2
        head_layer = 2

        subensemble_model = subensemble(
            model,
            num_heads=num_heads,
            head_layer=head_layer,
        )
        count_linear_original = count_layers(model, nnx.Linear)
        count_convolution_original = count_layers(model, nnx.Conv)
        count_sequential_original = count_layers(model, nnx.Sequential)

        assert isinstance(subensemble_model, nnx.Module)
        assert len(subensemble_model) == num_heads

        for member in subensemble_model:
            count_linear_subensemble = count_layers(member, nnx.Linear)
            count_convolutional_subensemble = count_layers(member, nnx.Conv)
            count_sequential_subensemble = count_layers(member, nnx.Sequential)
            assert count_linear_subensemble == count_linear_original
            assert count_convolutional_subensemble == count_convolution_original
            assert count_sequential_subensemble == count_sequential_original * 3

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

        count_linear_original = count_layers(model, nnx.Linear)
        count_sequential_original = count_layers(model, nnx.Sequential)
        count_convolutional_original = count_layers(model, nnx.Conv)
        count_sequential_original = count_layers(model, nnx.Sequential)

        count_sequential_subensemble = count_layers(subensemble_model, nnx.Sequential)

        assert isinstance(subensemble_model, nnx.List)
        assert len(subensemble_model) == num_heads
        assert count_sequential_subensemble == 3 * num_heads * count_sequential_original
        for member in subensemble_model:
            count_linear_subensemble = count_layers(member, nnx.Linear)
            count_convolutional_subensemble = count_layers(member, nnx.Conv)
            count_sequential_subensemble = count_layers(member, nnx.Sequential)
            assert count_linear_subensemble == count_linear_original * 2
            assert count_convolutional_subensemble == count_convolutional_original * 2


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

        head_member1 = subensemble_result[0].layers[-1]
        head_member2 = subensemble_result[1].layers[-1]

        params_head1 = jax.tree_util.tree_leaves(head_member1)
        params_head2 = jax.tree_util.tree_leaves(head_member2)

        assert not jnp.array_equal(params_head1[1], params_head2[1])

        for member in subensemble_result:
            head_layer = member.layers[-1]
            params_head = jax.tree_util.tree_leaves(head_layer)

            for original_params_last_layer_iterator, params_first_head_iterator in zip(
                original_params_last_layer[1],
                params_head[1],
                strict=False,
            ):
                assert not jnp.array_equal(original_params_last_layer_iterator, params_first_head_iterator)

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

        for member in subensemble_result:
            head_layer = member.layers[-1]
            params_head = jax.tree_util.tree_leaves(head_layer)
            for original_params_last_layer_iterator, params_head_iterator in zip(
                original_params_last_layer[1],
                params_head[1],
                strict=False,
            ):
                assert jnp.array_equal(original_params_last_layer_iterator, params_head_iterator)


class TestEdgeCases:
    """Tests for edge-case configurations."""

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
        head_layer = count_layers(flax_model_small_2d_2d, nnx.Linear) + 1

        with pytest.raises(
            ValueError,
            match=f"head_layer {head_layer} must be less than to {head_layer - 1}",
        ):
            subensemble(
                flax_model_small_2d_2d,
                num_heads=num_heads,
                head_layer=head_layer,
            )
