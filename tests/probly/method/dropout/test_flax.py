"""Test for flax dropout models."""

from __future__ import annotations

import pytest

from probly.method.dropout import dropout
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def test_shared_mask_not_implemented_for_flax(flax_model_small_2d_2d: nnx.Module) -> None:
    """shared_mask=True is only supported on the torch backend and raises for flax."""
    with pytest.raises(NotImplementedError, match="torch backend"):
        dropout(flax_model_small_2d_2d, p=0.5, shared_mask=True)


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "flax_model_small_2d_2d",
            "flax_conv_linear_model",
            "flax_regression_model_1d",
            "flax_regression_model_2d",
            "flax_dropout_model",
            "flax_dropconnect_model",
        ],
    )
    def test_fixtures(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        """Tests if a model incorporates a dropout layer correctly when a linear layer succeeds it.

        This function verifies that:
        - A dropout layer is added before each linear layer in the model, except for the last linear layer.
        - The structure of the model remains unchanged except for the added dropout layers.
        - Only the specified probability parameter is applied in dropout modifications.

        It performs counts and asserts to ensure the modified model adheres to expectations.

        Parameters:
            request: pytest.FixtureRequest, the request for a fixture.
            model_fixture: str, the name of the model fixture.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the dropout layer is not
            inserted correctly after linear layers.
        """
        p = 0.5
        model = request.getfixturevalue(model_fixture)

        modified_model = dropout(model, p)

        # count number of layer type layers in fixture model
        count_linear_original = count_layers(model, nnx.Linear)
        count_conv_original = count_layers(model, nnx.Conv)
        if isinstance(model.layers[0], nnx.Linear):
            count_linear_original_modifiable = count_layers(model, nnx.Linear) - 1
        else:
            count_linear_original_modifiable = count_layers(model, nnx.Linear)
        count_dropout_original = count_layers(model, nnx.Dropout)
        count_sequential_original = count_layers(model, nnx.Sequential)

        # count number of layer type layers in modified model
        count_linear_modified = count_layers(modified_model, nnx.Linear)
        count_conv_modified = count_layers(modified_model, nnx.Conv)
        count_dropout_modified = count_layers(modified_model, nnx.Dropout)
        count_sequential_modified = count_layers(modified_model, nnx.Sequential)

        # check that the model is modified as expected
        assert modified_model is not None
        assert isinstance(modified_model, type(model))
        assert count_conv_modified == count_conv_original
        assert count_dropout_modified == count_linear_original_modifiable + count_dropout_original
        assert count_linear_modified == count_linear_original
        assert count_sequential_original == count_sequential_modified

        # check p value in dropout layer
        if model_fixture != "flax_dropout_model":
            for m in nnx.iter_modules(model):
                if isinstance(m, nnx.Dropout):
                    assert m.rate == p

    def test_custom_model(self, flax_custom_model: nnx.Module) -> None:
        """Tests the custom model modification with added dropout layers."""
        p = 0.5
        model = dropout(flax_custom_model, p)

        count_linear_original = count_layers(flax_custom_model, nnx.Linear)

        count_linear_modified = count_layers(model, nnx.Linear)
        count_dropout_modified = count_layers(model, nnx.Dropout)

        # check if model type is correct and was modified as expected
        assert isinstance(model, type(flax_custom_model))
        assert not isinstance(model, nnx.Sequential)
        assert count_dropout_modified == count_linear_original - 1
        assert count_linear_modified == count_linear_original

        # check p value in dropconnect layer
        for m in nnx.iter_modules(flax_custom_model):
            if isinstance(m, nnx.Dropout):
                assert m.rate == p


class TestFirstLayer:
    """The first layer, which gets no dropout, is the first layer with parameters."""

    def test_leading_parameter_free_module_is_not_the_first_layer(self, flax_rngs: nnx.Rngs) -> None:
        base = nnx.Sequential(
            nnx.Dropout(rate=0.1, rngs=flax_rngs),
            nnx.Linear(4, 16, rngs=flax_rngs),
            nnx.relu,
            nnx.Linear(16, 3, rngs=flax_rngs),
        )

        model = dropout(base, p=0.25)

        layers = list(model.layers)
        assert len(layers) == 5
        assert isinstance(layers[0], nnx.Dropout)
        assert layers[0].rate == 0.1
        assert isinstance(layers[1], nnx.Linear)
        assert jnp.array_equal(layers[2](jnp.array([-1.0, 2.0])), jnp.array([0.0, 2.0]))  # the relu
        assert isinstance(layers[3], nnx.Dropout)
        assert layers[3].rate == 0.25
        assert isinstance(layers[4], nnx.Linear)
