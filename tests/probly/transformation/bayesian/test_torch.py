"""Tests for the bayesian transformation on PyTorch models."""

from __future__ import annotations

import pytest
import torch as th
from torch import nn

from probly.layers.torch import BayesConv2d, BayesLinear
from probly.transformation import bayesian

torch = pytest.importorskip("torch")


# --- Helper Functions for Testing ---


def inverse_softplus(x: th.Tensor) -> th.Tensor:
    """Compute the inverse softplus function for testing purposes (based on BayesLinear source)."""
    return th.log(th.exp(x) - 1)


def get_layer_count(model: nn.Module, layer_type: type) -> int:
    """Helper function to count specific layer types within a model."""
    count = 0
    for module in model.modules():
        if isinstance(module, layer_type):
            count += 1
    return count


# --- Test Cases ---


def test_bayesian_transformation_integrity(torch_model_small_2d_2d: nn.Module) -> None:
    """Tests the transformation of a PyTorch model into a Bayesian Predictor.

    This verifies that linear and conv2d layers are replaced, that the output
    shape is preserved, and that the model's train/eval modes function correctly.

    Parameters:
        torch_model_small_2d_2d (nn.Module): A small PyTorch model fixture
            containing Linear layers (based on runtime analysis).
    """
    # --- Setup ---
    # Use 2D input (Batch x Features) matching Linear(in_features=2)
    input_data = th.randn(1, 2)
    original_output = torch_model_small_2d_2d(input_data)
    expected_output_shape = original_output.shape

    # --- Transformation ---
    result = bayesian(torch_model_small_2d_2d)

    # 1. Transformation Check
    assert get_layer_count(result, BayesLinear) == 3
    assert get_layer_count(result, BayesConv2d) == 0
    assert get_layer_count(result, nn.Linear) == 0

    # 2. Shape Integrity Check
    transformed_output = result(input_data)
    assert transformed_output.shape == expected_output_shape

    # 3. Train/Eval Modes Check
    result.train()
    assert result.training
    result.eval()
    assert not result.training


def test_bayesian_parameter_override(torch_model_small_2d_2d: nn.Module) -> None:
    """Tests that parameters correctly override Bayesian layer defaults.

    Parameters:
        torch_model_small_2d_2d (nn.Module): A small PyTorch model fixture.
    """
    # --- Setup ---
    custom_prior_std = 10.0
    custom_posterior_std = 0.5

    # Calculate the expected value stored in weight_rho (rho = inverse_softplus(std))
    # We ensure the input is a tensor for inverse_softplus
    expected_rho_value = inverse_softplus(th.tensor(custom_posterior_std)).item()

    # --- Transformation with custom parameters ---
    result = bayesian(
        torch_model_small_2d_2d,
        prior_std=custom_prior_std,
        posterior_std=custom_posterior_std,
    )

    # 1. Parameter Check
    for module in result.modules():
        if isinstance(module, BayesLinear):
            # 1a. Check prior_std (stored in weight_prior_sigma buffer)
            # FIX: Use unique().item() to safely extract the single, repeated scalar from the tensor.
            assert module.weight_prior_sigma.data.unique().item() == pytest.approx(custom_prior_std)

            # 1b. Check posterior_std (stored in weight_rho parameter as rho)
            # FIX: Use unique().item() to safely extract the single, repeated scalar from the tensor.
            assert module.weight_rho.data.unique().item() == pytest.approx(expected_rho_value)

            # Stop after checking the first BayesLinear layer
            return


def test_bayesian_cloning_integrity(torch_model_small_2d_2d: nn.Module) -> None:
    """Tests transformation cloning integrity (base model must remain unchanged).

    Parameters:
        torch_model_small_2d_2d (nn.Module): A small PyTorch model fixture.
    """
    # --- Transformation ---
    result = bayesian(torch_model_small_2d_2d)

    # 1. Cloning Check
    # The result object must not be the same instance as the original.
    assert result is not torch_model_small_2d_2d

    # 2. State Check
    # The original model's training state must remain unchanged.
    original_state = torch_model_small_2d_2d.training
    result.train()
    # If cloning was successful, the state of the original model must be preserved
    assert torch_model_small_2d_2d.training == original_state
