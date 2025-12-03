"""Test for flax dropout models."""

from __future__ import annotations

import pytest
import jax 

from probly.transformation import subensemble
from tests.probly.flax_utils import count_layers

jax = pytest.importorskip("jax")
from jax import numpy as jnp # noqa: E402

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402

class TestGenerate:
    """Test class for subensemble generation."""

    def test_number_of_heads(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the subensemble transformation creates the correct number of heads."""

        num_heads = 4

        # default subensemble
        subensemble_result = subensemble(num_heads=num_heads, base=flax_model_small_2d_2d,reset_params=True)
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        # check tha backbone and heads are created correctly and heads have the correct structure
        assert backbone is not None
        assert heads is not None
        assert len(heads) == num_heads
        for head in heads:
            assert count_layers(head, nnx.Linear) == 1

    def test_number_of_heads_zero_head_layer(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the subensemble transformation creates the correct number of heads when head_layer is zero."""

        num_heads = 3
        # how many layers from the end should be considered as head
        head_layer = 1

        subensemble_result = subensemble(num_heads=num_heads, base=flax_model_small_2d_2d, head_layer=head_layer, reset_params=True)
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        nnx.display(subensemble_result)

        original_layers = count_layers(flax_model_small_2d_2d, nnx.Linear)
        modified_layers = count_layers(backbone, nnx.Linear)

        # check tha backbone is created correctly and no heads are created 
        assert backbone is not None
        assert heads is not None
        assert len(heads) == num_heads
        # check tha last layer is removed from backbone
        # assert original_layers ==  modified_layers - 1
    
    def test_number_of_heads_large_head_layer(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the subensemble transformation creates the correct number of heads when head_layer exceeds model depth."""

        num_heads = 2
        # how many layers from the end should be considered as head
        head_layer = 2

        # subensemble with input for head_layer
        subensemble_result = subensemble(num_heads=num_heads, base=flax_model_small_2d_2d,head_layer= head_layer, reset_params=True)
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        original_layers = count_layers(flax_model_small_2d_2d, nnx.Linear)
        modified_layers = count_layers(backbone, nnx.Linear)

        # check tha backbone is created correctly and heads have the correct structure
        assert backbone is not None
        assert heads is not None
        assert original_layers - 2 == modified_layers 
        assert len(heads) == num_heads
        for head in heads:
            assert count_layers(head, nnx.Linear) == head_layer
        
class TestReset:
    """Test class for parameter resetting."""

    def test_parameters_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the parameters of the model are reset correctly."""

        num_heads = 2

        original_params = jax.tree_util.tree_leaves(flax_model_small_2d_2d.layers[-1])

        subensemble_result = subensemble(num_heads=num_heads, base=flax_model_small_2d_2d, reset_params=True)
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        head1 = heads[0]
        modified_params_head1 = jax.tree_util.tree_leaves(head1)
       
        for head in heads:
            modified_params_last_layer = jax.tree_util.tree_leaves(head)
            
            for original_param, modified_param1 in zip(original_params[1], modified_params_last_layer[1]):
                assert not jnp.array_equal(original_param, modified_param1)
            
            for head1_param, modified_param2 in zip(modified_params_head1[1], modified_params_last_layer[1]):
                if head is head1:
                    continue
                assert not jnp.array_equal(head1_param, modified_param2)

    def test_parameters_not_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the parameters of the model are not reset when specified."""

        num_heads = 2

        original_params = jax.tree_util.tree_leaves(flax_model_small_2d_2d.layers[-1])

        subensemble_result = subensemble(num_heads=num_heads, base=flax_model_small_2d_2d, reset_params=False)
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        for head in heads:
            modified_params_last_layer = jax.tree_util.tree_leaves(head)
            for original_param, modified_param in zip(original_params[1], modified_params_last_layer[1]):
                assert jnp.array_equal(original_param, modified_param)

    def test_forward_pass_shape(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the forward pass works correctly after subensemble transformation."""

        num_heads = 2

        subensemble_result = subensemble(num_heads=num_heads, base=flax_model_small_2d_2d, reset_params=True)
        backbone = subensemble_result[0]
        heads = subensemble_result[1]

        # create dummy input
        x = jnp.ones((1, 2))

        # forward pass through backbone
        if backbone is not None:
            backbone_output = backbone(x)
        else:
            backbone_output = x

        # forward pass through each head
        for head in heads:
            head_output = head(backbone_output)
            assert head_output.shape == (1, 2)
