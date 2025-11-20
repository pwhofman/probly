"""Tests for evidential regression transformation.

This module tests the probly.transformation.evidential.regression package,
which transforms traditional neural networks into evidential regression models
for uncertainty quantification.
"""

import pytest
import torch
from torch import nn

from probly.transformation.evidential.regression import evidential_regression


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple three-layer neural network for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )


@pytest.fixture
def mlp_model():
    """Create a multi-layer perceptron (MLP) for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )


@pytest.fixture
def single_layer_model():
    """Create a single-layer model for edge case testing."""
    return nn.Linear(10, 1)


@pytest.fixture
def sample_input():
    """Create sample input data: 5 samples, each with 10 dimensions."""
    return torch.randn(5, 10)


# ============================================================================
# Test 1: Basic Transformation
# ============================================================================

def test_transformation_completes_successfully(simple_model):
    """Test that the transformation completes without errors.
    
    Verify that the evidential_regression() function runs successfully
    and returns a valid PyTorch model.
    """
    transformed_model = evidential_regression(simple_model)
    
    assert transformed_model is not None, \
        "Transformed model should not be None"
    assert isinstance(transformed_model, nn.Module), \
        "Transformed model should be a PyTorch Module"


# ============================================================================
# Test 2: Layer Replacement Verification
# ============================================================================

def test_last_linear_layer_replaced_with_nig(simple_model):
    """Test that the last Linear layer is replaced with NormalInverseGammaLinear.
    
    This is the core functionality of evidential regression.
    Verify that the type of the last layer is correct.
    """
    from probly.layers.torch import NormalInverseGammaLinear
    
    transformed_model = evidential_regression(simple_model)
    layers = list(transformed_model.children())
    
    assert isinstance(layers[-1], NormalInverseGammaLinear), \
        "Last layer should be NormalInverseGammaLinear"


# ============================================================================
# Test 3: Single Layer Replacement
# ============================================================================

def test_only_one_layer_is_replaced(mlp_model):
    """Test that only one layer is replaced, not all Linear layers.
    
    Verify that the transformation does not over-replace.
    In a multi-layer network, only the last Linear layer should be replaced.
    """
    from probly.layers.torch import NormalInverseGammaLinear
    
    transformed_model = evidential_regression(mlp_model)
    
    nig_count = sum(
        1 for module in transformed_model.modules() 
        if isinstance(module, NormalInverseGammaLinear)
    )
    
    assert nig_count == 1, \
        f"Should have exactly 1 NormalInverseGammaLinear layer, but found {nig_count}"


# ============================================================================
# Test 4: Forward Pass Functionality
# ============================================================================

def test_forward_pass_executes_without_error(simple_model, sample_input):
    """Test that the forward pass runs successfully.
    
    Verify that the transformed model can actually be used
    and produces some output.
    """
    transformed_model = evidential_regression(simple_model)
    output = transformed_model(sample_input)
    
    assert output is not None, "Output should not be None"
    
    # Check if output is a tensor or dict containing tensors
    if isinstance(output, torch.Tensor):
        assert torch.all(torch.isfinite(output)), \
            "All output values should be finite"
    elif isinstance(output, dict):
        # If it's a dict, check that it contains at least one tensor
        has_tensor = any(isinstance(v, torch.Tensor) for v in output.values())
        assert has_tensor, "Output dict should contain at least one tensor"


# ============================================================================
# Test 5: Original Model Preservation
# ============================================================================

def test_original_model_remains_unchanged(simple_model):
    """Test that the original model is not modified.
    
    Verify that the transformation creates a copy rather than modifying
    the original model. This ensures users can keep the original model.
    """
    original_last_layer = list(simple_model.children())[-1]
    original_layer_type = type(original_last_layer)
    
    _ = evidential_regression(simple_model)
    
    current_last_layer = list(simple_model.children())[-1]
    current_layer_type = type(current_last_layer)
    
    assert current_layer_type == original_layer_type, \
        "Original model should not be modified"
    assert isinstance(current_last_layer, nn.Linear), \
        "Original model's last layer should still be Linear"


# ============================================================================
# Test 6: Output Structure Verification
# ============================================================================

def test_output_produces_valid_data(simple_model, sample_input):
    """Test that the output has valid structure and dimensions.
    
    The output should be either:
    - A tensor with appropriate dimensions
    - A dict containing tensors with appropriate dimensions
    """
    transformed_model = evidential_regression(simple_model)
    output = transformed_model(sample_input)
    
    # Extract tensor from output
    if isinstance(output, torch.Tensor):
        output_tensor = output
    elif isinstance(output, dict):
        # Find the first tensor in the dict
        output_tensor = None
        for value in output.values():
            if isinstance(value, torch.Tensor):
                output_tensor = value
                break
        assert output_tensor is not None, "Dict should contain at least one tensor"
    else:
        raise AssertionError(f"Unexpected output type: {type(output)}")
    
    # Check basic properties
    assert output_tensor.shape[0] == 5, "Batch dimension should be 5"
    assert len(output_tensor.shape) == 2, "Output should be 2-dimensional"
    assert torch.all(torch.isfinite(output_tensor)), "All values should be finite"


# ============================================================================
# Test 7: Gradient Computation
# ============================================================================

def test_model_supports_gradient_computation(simple_model, sample_input):
    """Test that the model supports gradient computation.
    
    Verify that the transformed model can be used for training
    by checking that gradients can be computed.
    """
    transformed_model = evidential_regression(simple_model)
    
    # Zero gradients
    transformed_model.zero_grad()
    
    # Forward pass
    output = transformed_model(sample_input)
    
    # Extract tensor and compute loss
    if isinstance(output, torch.Tensor):
        loss = output.sum()
    elif isinstance(output, dict):
        # Find first tensor for loss computation
        for value in output.values():
            if isinstance(value, torch.Tensor):
                loss = value.sum()
                break
    
    # Backward pass
    loss.backward()
    
    # Check that at least some gradients are computed
    grad_count = sum(
        1 for param in transformed_model.parameters() 
        if param.requires_grad and param.grad is not None
    )
    
    assert grad_count > 0, \
        "At least some parameters should have gradients computed"


# ============================================================================
# Test 8: Edge Case - Single Layer Model
# ============================================================================

def test_single_layer_model_edge_case(single_layer_model):
    """Test transformation on a single-layer model (edge case).
    
    Verify that even a single Linear layer can be correctly transformed.
    """
    from probly.layers.torch import NormalInverseGammaLinear
    
    transformed_model = evidential_regression(single_layer_model)
    
    assert isinstance(transformed_model, NormalInverseGammaLinear), \
        "Single Linear layer should be replaced with NormalInverseGammaLinear"


# ============================================================================
# Test 9: Device Preservation
# ============================================================================

def test_device_preservation_cpu(simple_model):
    """Test that CPU device is correctly preserved.
    
    Verify that the transformation does not change the model's device location.
    """
    simple_model = simple_model.cpu()
    transformed_model = evidential_regression(simple_model)
    
    for param in transformed_model.parameters():
        assert param.device.type == 'cpu', \
            "Parameters should remain on CPU"


# ============================================================================
# Test 10: Different Model Architectures
# ============================================================================

def test_transformation_works_with_different_architectures():
    """Test that transformation works with various network architectures.
    
    Verify that the transformation is robust and works with
    different types of neural network architectures.
    """
    from probly.layers.torch import NormalInverseGammaLinear
    
    # Test with very simple model (2 layers)
    simple = nn.Sequential(nn.Linear(5, 10), nn.Linear(10, 1))
    transformed_simple = evidential_regression(simple)
    assert isinstance(list(transformed_simple.children())[-1], NormalInverseGammaLinear)
    
    # Test with model containing batch norm
    with_bn = nn.Sequential(
        nn.Linear(10, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    transformed_bn = evidential_regression(with_bn)
    assert isinstance(list(transformed_bn.children())[-1], NormalInverseGammaLinear)
    
    # Test with model containing dropout
    with_dropout = nn.Sequential(
        nn.Linear(10, 20),
        nn.Dropout(0.5),
        nn.Linear(20, 1)
    )
    transformed_dropout = evidential_regression(with_dropout)
    assert isinstance(list(transformed_dropout.children())[-1], NormalInverseGammaLinear)
    
