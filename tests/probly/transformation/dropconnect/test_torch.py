"""Test for torch dropconnect models."""

from __future__ import annotations

import pytest

from probly.transformation import dropconnect
from tests.probly.torch_utils import count_layers
from probly.layers.torch import DropConnectLinear

torch = pytest.importorskip("torch")

from torch import nn 

class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if every linear layer but the first in a linear model gets changed to a drop connect linear layer.

        This function verifies that:
        - Every linear layer but the first is exchanged for a linear drop connect layer.
        - The structure of the model remains unchanged.

        Parameters:
            torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the architecture of the model got changed unexpectedly or if an incorrect number of
            drop connect layers was inserted.
        """

        # P-Value for drop connect layer
        p = 0.5
        # transformed base model
        model = dropconnect(torch_model_small_2d_2d, p)
        
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        count_drop_connect_original = count_layers(torch_model_small_2d_2d, DropConnectLinear)
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        count_linear_modified = count_layers(model, nn.Linear)
        count_drop_connect_modified = count_layers(model, DropConnectLinear)
        count_sequential_modified = count_layers(model, nn.Sequential)

        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_drop_connect_modified == count_linear_original - 1
        assert count_drop_connect_original == 0
        assert count_linear_modified == 1
        assert count_sequential_original == count_sequential_modified

    def test_regression_network_1d(self, torch_regression_model_1d: nn.Sequential) -> None:
        """Tests if every linear layer but the first in a linear regression model gets changed to a drop connect linear layer.

        This function verifies that:
        - Every linear layer but the first is exchanged for a linear drop connect layer.
        - The structure of the model remains unchanged.

        Parameters:
            torch_regression_model_1d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the architecture of the model got changed unexpectedly or if an incorrect number of
            drop connect layers was inserted.
        """

        # P-Value for drop connect layer
        p = 0.5
        # transformed base model
        model = dropconnect(torch_regression_model_1d, p)

        count_linear_original = count_layers(torch_regression_model_1d, nn.Linear)
        count_drop_connect_original = count_layers(torch_regression_model_1d, DropConnectLinear)
        count_sequential_original = count_layers(torch_regression_model_1d, nn.Sequential)

        count_linear_modified = count_layers(model, nn.Linear)
        count_drop_connect_modified = count_layers(model, DropConnectLinear)
        count_sequential_modified = count_layers(model, nn.Sequential)

        assert model is not None
        assert isinstance(model, type(torch_regression_model_1d))
        assert count_drop_connect_modified == count_linear_original - 1
        assert count_drop_connect_original == 0
        assert count_linear_modified == 1
        assert count_sequential_original == count_sequential_modified

    def test_regression_network_2d(self, torch_regression_model_2d: nn.Sequential) -> None:
        """Tests a torch_regression_model_2d in the same way as the orch_regression_model_1d"""

        # P-Value for drop connect layer
        p = 0.5
        # transformed base model
        model = dropconnect(torch_regression_model_2d, p)

        count_linear_original = count_layers(torch_regression_model_2d, nn.Linear)
        count_drop_connect_original = count_layers(torch_regression_model_2d, DropConnectLinear)
        count_sequential_original = count_layers(torch_regression_model_2d, nn.Sequential)

        count_linear_modified = count_layers(model, nn.Linear)
        count_drop_connect_modified = count_layers(model, DropConnectLinear)
        count_sequential_modified = count_layers(model, nn.Sequential)

        assert model is not None
        assert isinstance(model, type(torch_regression_model_2d))
        assert count_drop_connect_modified == count_linear_original - 1
        assert count_drop_connect_original == 0
        assert count_linear_modified == 1
        assert count_sequential_original == count_sequential_modified

    def test_dropout_network(self, torch_dropout_model: nn.Sequential) -> None:
        """Tests if every linear layer but the first in a model with dropout layers gets changed to a drop connect linear layer.

        This function verifies that:
        - Every linear layer but the first is exchanged for a linear drop connect layer.
        - Every dropout layer remains unchanged.
        - The structure of the model remains unchanged.

        Parameters:
            torch_dropout_model: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the architecture of the model got changed unexpectedly or if an incorrect number of
            drop connect layers was inserted. And if the number of dropout layers changes.
        """

        # P-Value for drop connect layer
        p = 0.5
        # transformed base model
        model = dropconnect(torch_dropout_model, p)

        count_linear_original = count_layers(torch_dropout_model, nn.Linear)
        count_drop_connect_original = count_layers(torch_dropout_model, DropConnectLinear)
        count_dropout_original = count_layers(torch_dropout_model, nn.Dropout)
        count_sequential_original = count_layers(torch_dropout_model, nn.Sequential)

        count_linear_modified = count_layers(model, nn.Linear)
        count_drop_connect_modified = count_layers(model, DropConnectLinear)
        count_dropout_modified = count_layers(model, nn.Dropout)
        count_sequential_modified = count_layers(model, nn.Sequential)

        assert model is not None
        assert isinstance(model, type(torch_dropout_model))
        assert count_drop_connect_modified == count_linear_original - 1
        assert count_drop_connect_original == 0
        assert count_linear_modified == 1
        assert count_dropout_modified == count_dropout_original
        assert count_sequential_original == count_sequential_modified

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom torch model"""
        # P-Value for drop connect layer
        p = 0.5
        # transformed base model
        model = dropconnect(torch_custom_model, p)

        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)


class TestPValues:
    
    def test_linear_network_p_values(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if the drop connect layers have the correct p-value assigned to them
        
        Parameters:
            torch_model_small_2d_2d: The torch model to be tested for integration

        Raises:
            AssertionError: If the p-value in a Dropout layer does not match the expected value.
        """

        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert m.p == p

    def test_regression_network_1d_p_values(self, torch_regression_model_1d: nn.Sequential) -> None:
        """Tests if the drop connect layers have the correct p-value assigned to them
        
        Parameters:
            torch_regression_model_1d: The torch model to be tested for integration

        Raises:
            AssertionError: If the p-value in a Dropout layer does not match the expected value.
        """

        p = 0.5
        model = dropconnect(torch_regression_model_1d, p)

        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert m.p == p

    def test_regression_network_2d_p_values(self, torch_regression_model_2d: nn.Sequential) -> None:
        """Tests if the drop connect layers have the correct p-value assigned to them
        
        Parameters:
            torch_regression_model_2d: The torch model to be tested for integration

        Raises:
            AssertionError: If the p-value in a Dropout layer does not match the expected value.
        """

        p = 0.5
        model = dropconnect(torch_regression_model_2d, p)

        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert m.p == p

    def test_dropout_network_p_values(self, torch_dropout_model: nn.Sequential) -> None:
        """Tests if the drop connect layers have the correct p-value assigned to them
        
        Parameters:
            torch_dropout_model: The torch model to be tested for integration

        Raises:
            AssertionError: If the p-value in a Dropout layer does not match the expected value.
        """
        
        p = 0.5
        model = dropconnect(torch_dropout_model, p)

        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert m.p == p