"""Test for flax dropout models."""

from __future__ import annotations

import pytest

from probly.transformation import dropout
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_with_first_linear(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if a model incorporates a dropout layer correctly when a linear layer succeeds it.

        This function verifies that:
        - A dropout layer is added before each linear layer in the model, except for the last linear layer.
        - The structure of the model remains unchanged except for the added dropout layers.
        - Only the specified probability parameter is applied in dropout modifications.

        It performs counts and asserts to ensure the modified model adheres to expectations.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the dropout layer is not
            inserted correctly after linear layers.
        """
        p = 0.5
        model = dropout(flax_model_small_2d_2d, p)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_model_small_2d_2d, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_model_small_2d_2d, nnx.Sequential)

        # count number of nnx.Linear layers in modified model
        count_linear_modified = count_layers(model, nnx.Linear)
        # count number of nnx.Dropout layers in modified model
        count_dropout_modified = count_layers(model, nnx.Dropout)
        # count number of nnx.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nnx.Sequential)

        # check that the model is not modified except for the dropout layer
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert (count_linear_original - 1) == count_dropout_modified
        assert count_linear_modified == count_linear_original
        assert count_dropout_original == 0
        assert count_sequential_original == count_sequential_modified

    def test_convolutional_network(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests the convolutional neural network modification with added dropout layers.

        This function evaluates whether the given convolutional neural network model
        has been correctly modified to include dropout layers without altering the
        number of other components such as linear, sequential, or convolutional layers.

        Parameters:
            flax_conv_linear_model: The original convolutional neural network model to be tested.

        Raises:
            AssertionError: If the modified model deviates in structure other than
            the addition of dropout layers or does not meet the expected constraints.
        """
        p = 0.5
        model = dropout(flax_conv_linear_model, p)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_conv_linear_model, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_conv_linear_model, nnx.Sequential)
        # count number of nnx.Conv2d layers in original model
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv)

        # count number of nnx.Linear layers in modified model
        count_linear_modified = count_layers(model, nnx.Linear)
        # count number of nnx.Dropout layers in modified model
        count_dropout_modified = count_layers(model, nnx.Dropout)
        # count number of nnx.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nnx.Sequential)
        # count number of nnx.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nnx.Conv)

        # check that the model is not modified except for the dropout layer
        assert model is not None
        assert isinstance(model, type(flax_conv_linear_model))
        assert count_linear_original == count_dropout_modified
        assert count_linear_original == count_linear_modified
        assert count_dropout_original == 0
        assert count_sequential_original == count_sequential_modified
        assert count_conv_original == count_conv_modified

    def test_custom_network(self, flax_custom_model: nnx.Module) -> None:
        """Tests the custom model modification with added dropout layers."""
        p = 0.5
        model = dropout(flax_custom_model, p)

        # check if model type is correct
        assert isinstance(model, type(flax_custom_model))
        assert not isinstance(model, nnx.Sequential)

    @pytest.mark.skip(reason="Not yet implemented")
    def test_dropout_model(self, flax_dropout_model: nnx.Sequential) -> None:
        """Tests the dropout model modification if dropout already exists."""
        p = 0.2
        model = dropout(flax_dropout_model, p)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_dropout_model, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_dropout_model, nnx.Dropout)
        # count number of nnx.Dropout layers in modified model
        count_dropout_modified = count_layers(model, nnx.Dropout)

        # check that model has no duplicate dropout layers
        assert count_dropout_original == 1
        assert count_linear_original == 2
        assert (count_linear_original - 1) == count_dropout_modified

        # check p value in dropout layer
        for m in model.layers:
            if isinstance(m, nnx.Dropout):
                assert m.rate == p


class TestPValues:
    """Test class for p-value tests."""

    def test_linear_network_p_value(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests the Dropout layer's p-value in a given neural network model.

        This function verifies that a Dropout layer inside the provided neural network
        model has the expected p-value after applying the dropout transformation. The
        p-value represents the probability of an element being zeroed during training.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested for integration

        Raises:
            AssertionError: If the p-value in a Dropout layer does not match the expected value.
        """
        p = 0.5
        model = dropout(flax_model_small_2d_2d, p)

        # check p value in dropout layer
        for m in model.layers:
            if isinstance(m, nnx.Dropout):
                assert m.rate == p

    def test_conv_network_p_value(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """This function tests whether the dropout layer in the convolutional model has the correct probability value.

        Arguments:
            flax_conv_linear_model: A sequential model containing convolutional and linear layers.

        Raises:
            AssertionError: If the probability value in any dropout layer does not match the expected value.
        """
        p = 0.2
        model = dropout(flax_conv_linear_model, p)

        # check p value in dropout layer
        for m in model.layers:
            if isinstance(m, nnx.Dropout):
                assert m.rate == p
