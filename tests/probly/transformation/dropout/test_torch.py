"""Test for torch dropout models."""

from __future__ import annotations

import pytest

from probly.transformation import dropout

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


def test_linear_network_with_first_linear(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Tests if a model incorporates a dropout layer correctly when a linear layer succeeds it.

    This function verifies that:
    - A dropout layer is added before each linear layer in the model, except for the last linear layer.
    - The structure of the model remains unchanged except for the added dropout layers.
    - Only the specified probability parameter is applied in dropout modifications.

    It performs counts and asserts to ensure the modified model adheres to expectations.

    Parameters:
        torch_model_small_2d_2d: The torch model to be tested, specified as a sequential model.

    Raises:
        AssertionError: If the structure of the model differs in an unexpected manner or if the dropout layer is not
        inserted correctly after linear layers.
    """
    p = 0.5
    model = dropout(torch_model_small_2d_2d, p)

    # count number of nn.Linear layers in original model
    count_linear_original = sum(1 for m in torch_model_small_2d_2d.modules() if isinstance(m, nn.Linear))
    # count number of nn.Dropout layers in original model
    count_dropout_original = sum(1 for m in torch_model_small_2d_2d.modules() if isinstance(m, nn.Dropout))
    # count number of nn.Sequential layers in original model
    count_sequential_original = sum(1 for m in torch_model_small_2d_2d.modules() if isinstance(m, nn.Sequential))

    # count number of nn.Dropout layers in modified model
    count_dropout_modified = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
    # count number of nn.Linear layers in modified model
    count_linear_modified = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    # count number of nn.Sequential layers in modified model
    count_sequential_modified = sum(1 for m in model.modules() if isinstance(m, nn.Sequential))

    # check that the model is not modified except for the dropout layer
    assert model is not None
    assert isinstance(model, nn.Module)
    assert (count_linear_original - 1) == count_dropout_modified
    assert count_linear_modified == count_linear_original
    assert count_dropout_original == 0
    assert (
        count_sequential_original == count_sequential_modified
    )  # todos: Is that useful with our nn_traverser cleaning models?


def test_linear_network_p_value(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Tests the Dropout layer's p-value in a given neural network model.

    This function verifies that a Dropout layer inside the provided neural network
    model has the expected p-value after applying the dropout transformation. The
    p-value represents the probability of an element being zeroed during training.

    Parameters:
        torch_model_small_2d_2d: The torch model to be tested for integration

    Raises:
        AssertionError: If the p-value in a Dropout layer does not match the expected value.
    """
    p = 0.5
    model = dropout(torch_model_small_2d_2d, p)

    # check p value in dropout layer
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            assert m.p == p


def test_convolutional_network(torch_conv_linear_model: nn.Sequential) -> None:
    """Tests the convolutional neural network modification with added dropout layers.

    This function evaluates whether the given convolutional neural network model
    has been correctly modified to include dropout layers without altering the
    number of other components such as linear, sequential, or convolutional layers.

    Parameters:
        torch_conv_linear_model: The original convolutional neural network model to be tested.

    Raises:
        AssertionError: If the modified model deviates in structure other than
        the addition of dropout layers or does not meet the expected constraints.
    """
    p = 0.5
    model = dropout(torch_conv_linear_model, p)

    # count number of nn.Linear layers in original model
    count_linear_original = sum(1 for m in torch_conv_linear_model.modules() if isinstance(m, nn.Linear))
    # count number of nn.Dropout layers in original model
    count_dropout_original = sum(1 for m in torch_conv_linear_model.modules() if isinstance(m, nn.Dropout))
    # count number of nn.Sequential layers in original model
    count_sequential_original = sum(1 for m in torch_conv_linear_model.modules() if isinstance(m, nn.Sequential))
    # count number of nn.Conv2d layers in original model
    count_conv_original = sum(1 for m in torch_conv_linear_model.modules() if isinstance(m, nn.Conv2d))

    # count number of nn.Dropout layers in modified model
    count_dropout_modified = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
    # count number of nn.Linear layers in modified model
    count_linear_modified = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    # count number of nn.Sequential layers in modified model
    count_sequential_modified = sum(1 for m in model.modules() if isinstance(m, nn.Sequential))
    # count number of nn.Conv2d layers in modified model
    count_conv_modified = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))

    # check that the model is not modified except for the dropout layer
    assert model is not None
    assert isinstance(model, nn.Module)
    assert count_linear_original == count_dropout_modified
    assert count_linear_original == count_linear_modified
    assert count_dropout_original == 0
    assert (
        count_sequential_original == count_sequential_modified
    )  # todos: Is that useful with our nn_traverser cleaning models?
    assert count_conv_original == count_conv_modified


def test_conv_network_p_value(torch_conv_linear_model: nn.Sequential) -> None:
    """This function tests whether the dropout layer in the convolutional model has the correct probability value.

    Arguments:
        torch_conv_linear_model: A sequential model containing convolutional and linear layers.

    Raises:
        AssertionError: If the probability value in any dropout layer does not match the expected value.
    """
    p = 0.2
    model = dropout(torch_conv_linear_model, p)

    # check p value in dropout layer
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            assert m.p == p


# todos: possible other scenarios:
# - models which already have Dropouts
# - models with multiple Sequentials
# - p value not between 0 and 1 -> currently possible but not intended
