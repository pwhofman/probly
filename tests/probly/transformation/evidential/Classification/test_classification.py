"""Tests for evidential classification transformation.

This module tests the probly.transformation.evidential.classification package,
which transforms traditional neural networks into evidential classification models
for uncertainty quantification.
"""

import pytest
import torch
from torch import nn

from probly.transformation import evidential_classification


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple three-layer neural network for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3)  # 3 classes
    )


@pytest.fixture
def mlp_model():
    """Create a multi-layer perceptron (MLP) for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 4)  # 4 classes
    )


@pytest.fixture
def single_layer_model():
    """Create a single-layer model for edge case testing."""
    return nn.Linear(10, 2)  # 2 classes


@pytest.fixture
def sample_input():
    """Create sample input data: 5 samples, each with 10 dimensions."""
    return torch.randn(5, 10)


# ============================================================================
# Test 1: Basic Transformation
# ============================================================================

def test_transformation_completes_successfully(simple_model):
    """Test that the transformation completes without errors.
    
    Verify that the evidential_classification() function runs successfully
    and returns a valid PyTorch model.
    """
    transformed_model = evidential_classification(simple_model)

    assert transformed_model is not None, \
        "Transformed model should not be None"
    assert isinstance(transformed_model, nn.Module), \
        "Transformed model should be a PyTorch Module"


# ============================================================================
# Test 2: Layer Replacement Verification
# ============================================================================

def test_last_linear_layer_replaced_with_softplus(simple_model):
    """Test that the last Linear layer is replaced with nn.Softplus.
    
    This is the core functionality of evidential classification.
    Verify that the type of the last layer is correct.
    """

    transformed_model = evidential_classification(simple_model)
    layers = list(transformed_model.children())

    assert isinstance(layers[-1], nn.Softplus), \
        "Last layer should be nn.Softplus"


# ============================================================================
# Test 3: Single Layer Replacement
# ============================================================================

def test_only_one_layer_is_replaced(mlp_model):
    """Test that only one layer is replaced, not all Linear layers.
    
    Verify that the transformation does not over-replace.
    In a multi-layer network, only the last Linear layer should be replaced.
    """

    transformed_model = evidential_classification(mlp_model)

    softplus_count = sum(
        1 for module in transformed_model.modules() 
        if isinstance(module, nn.Softplus)
    )

    assert softplus_count == 1, \
        f"Should have exactly 1 nn.Softplus layer, but found {softplus_count}"

# ============================================================================
# Test 4: Edge Case - Single Layer Model
# ============================================================================

def test_single_layer_model_edge_case(single_layer_model):
    """Test transformation on a single-layer model (edge case).
    
    Verify that even a single Linear layer can be correctly transformed.
    """

    transformed_model = evidential_classification(single_layer_model)

    # Check that the final layer is Softplus after transformation
    last_layer = list(transformed_model.children())[-1]

    assert isinstance(last_layer, nn.Softplus), \
        "Single Linear layer should be replaced with nn.Softplus"


# ============================================================================
# Test 5: Device Preservation
# ============================================================================

def test_device_preservation_cpu(simple_model):
    """Test that CPU device is correctly preserved.
    
    Verify that the transformation does not change the model's device location.
    """
    simple_model = simple_model.cpu()
    transformed_model = evidential_classification(simple_model)

    for param in transformed_model.parameters():
        assert param.device.type == 'cpu', \
            "Parameters should remain on CPU"

# ============================================================================
# Test 6: Softplus Activation Verification
# ============================================================================

def test_output_has_softplus_activation(simple_model, sample_input):
    """Test that the output has softplus activation applied.
    
    Verify that the evidential classification model applies softplus
    activation to ensure positive concentration parameters.
    """
    transformed_model = evidential_classification(simple_model)
    output = transformed_model(sample_input)

    # Extract the main output tensor
    if isinstance(output, torch.Tensor):
        output_tensor = output
    elif isinstance(output, dict):
        # Look for concentration parameters or main output
        output_tensor = None
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and ('concentration' in key or 'evidence' in key or 'alpha' in key):
                output_tensor = value
                break
        if output_tensor is None:
            # Fallback to first tensor
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    output_tensor = value
                    break

    # Check that all values are positive (due to softplus)
    assert torch.all(output_tensor >= 0), \
        "All output values should be non-negative due to softplus activation"

