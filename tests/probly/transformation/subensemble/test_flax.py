"""Test for flax dropout models."""

from __future__ import annotations

import jax
from jax import numpy as jnp
import pytest

from probly.transformation import subensemble
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


class TestGenerate:
    """Test class for subensemble generation."""

    def test_number_of_heads(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the subensemble transformation creates the correct number of heads."""
        num_heads = 4

        # default subensemble
        subensemble_result = subensemble(flax_model_small_2d_2d, num_heads=num_heads, reset_params=True)
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        original_layers = count_layers(flax_model_small_2d_2d, nnx.Linear)
        backbone_layers = count_layers(backbone, nnx.Linear)

        # check tha backbone and heads are created correctly and heads have the correct structure
        assert backbone is type(nnx.Sequential)
        assert heads is type(nnx.List)
        assert len(heads) == num_heads
        assert original_layers - 1 == backbone_layers
        for head in heads:
            assert count_layers(head, nnx.Linear) == 1

    def test_number_of_heads_large_head_layer(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the subensemble transformation creates the correct number and larger head_layer."""
        num_heads = 2
        # how many layers from the end should be considered as head
        head_layer = 2

        # subensemble with input for head_layer
        subensemble_result = subensemble(
            flax_model_small_2d_2d,
            num_heads=num_heads,
            head_layer=head_layer,
            reset_params=True,
        )
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        original_layers = count_layers(flax_model_small_2d_2d, nnx.Linear)
        backbone_layers = count_layers(backbone, nnx.Linear)

        # check tha backbone is created correctly and heads have the correct structure
        assert isinstance(backbone, nnx.Sequential)
        assert isinstance(heads, nnx.List)
        assert original_layers - 2 == backbone_layers
        assert len(heads) == num_heads
        for head in heads:
            assert count_layers(head, nnx.Linear) == head_layer


class TestReset:
    """Test class for parameter resetting."""

    def test_parameters_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the parameters of the model are reset correctly."""
        num_heads = 2

        original_params_last_layer = jax.tree_util.tree_leaves(flax_model_small_2d_2d.layers[-1])

        subensemble_result = subensemble(flax_model_small_2d_2d, num_heads=num_heads, reset_params=True)
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

    def test_parameters_not_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the parameters of the model are not reset when specified."""
        num_heads = 2

        original_params_last_layer = jax.tree_util.tree_leaves(flax_model_small_2d_2d.layers[-1])

        subensemble_result = subensemble(flax_model_small_2d_2d, num_heads=num_heads, reset_params=False)
        heads = subensemble_result[1]

        for head in heads:
            params_head = jax.tree_util.tree_leaves(head)
            for original_params_last_layer_iterator, params_head_iterator in zip(
                original_params_last_layer[1],
                params_head[1],
                strict=False,
            ):
                assert jnp.array_equal(original_params_last_layer_iterator, params_head_iterator)

    def test_forward_pass_shape(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the forward pass works correctly after subensemble transformation."""
        num_heads = 2

        subensemble_result = subensemble(num_heads=num_heads, base=flax_model_small_2d_2d, reset_params=True)
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        # create dummy input
        x = jnp.ones((1, 2))

        # forward pass through backbone

        backbone_output = backbone(x) if backbone is not None else x

        # forward pass through each head
        for head in heads:
            head_output = head(backbone_output)
            assert head_output.shape == x.shape
