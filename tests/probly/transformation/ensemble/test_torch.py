"""Test for torch ensemble models.."""

from __future__ import annotations
import pytest
from probly.transformation import ensemble
from probly.transformation.ensemble.torch import generate_torch_ensemble, _reset_copy
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn

class TestNetworkArchitectures:
    """Test different Network Architectures for torch ensemble models."""



def test_linear_network_with_first_linear(torch_model_small_2d_2d: nn.Sequential) -> None:
    n_memb = 1
    model = ensemble(torch_model_small_2d_2d, n_members=n_memb)

    count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
    count_linear_modified = count_layers(model, nn.Linear)
    count_dropout_modified = count_layers(model, nn.Dropout)

    # test für Ensemble
    assert model is not None
    assert isinstance(model, nn.ModuleList)
    assert len(model) == n_memb
    assert count_linear_modified == count_linear_original  # gleiche Anzahl Linear-Layer
    assert count_dropout_modified == 0  # keine Dropout-Layer hinzugefügt



def test_convolutional_network(torch_conv_linear_model: nn.Sequential) -> None:

    n_memb = 1
    model = ensemble(torch_conv_linear_model, n_members=n_memb)

    count_linear_orginal = count_layers(torch_conv_linear_model, nn.Linear)
    count_linear_modified = count_layers(model, nn.Linear)
    count_dropout_modified = count_layers(model, nn.Dropout)
    count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
    count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)
    count_conv_modified = count_layers(model, nn.Conv2d)
    count_sequential_modified = count_layers(model, nn.Sequential)

    #tests
    assert model is not None
    assert isinstance(model, nn.Module)
    assert len(model) == n_memb
    assert count_linear_orginal == count_linear_modified
    assert count_dropout_modified == 0
    assert count_sequential_modified == count_sequential_original
    assert count_conv_modified == count_conv_original

def test_custom_network(torch_custom_model: nn.Module) -> None:
    n_memb = 1
    model = ensemble(torch_custom_model, n_members=n_memb)

    count_linear_original = count_layers(torch_custom_model, nn.Linear)
    count_linear_modified = count_layers(model, nn.Linear)
    count_dropout_modified = count_layers(model, nn.Dropout)
    count_sequential_original = count_layers(torch_custom_model, nn.Sequential)
    count_conv_original = count_layers(torch_custom_model, nn.Conv2d)
    count_conv_modified = count_conv_modified = count_layers(model, nn.Conv2d)
    count_sequential_modified = count_layers(torch_custom_model, nn.Sequential)


    assert model is not None
    assert isinstance(model, nn.Module)
    assert len(model) == n_memb
    assert count_linear_modified == count_linear_original
    assert count_dropout_modified == 0
    assert count_sequential_modified == count_sequential_original
    assert count_conv_original == count_conv_modified




def test_ensemble_returns_module_list(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Test that ensemble returns a ModuleList."""
    n_members = 3
    model = ensemble(torch_model_small_2d_2d, n_members=n_members)

    assert isinstance(model, nn.ModuleList)


def test_ensemble_correct_number_of_members(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Test that ensemble creates correct number of members."""
    n_members = 5
    model = ensemble(torch_model_small_2d_2d, n_members=n_members)

    assert len(model) == n_members


def test_ensemble_members_have_same_type(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Test that each ensemble member has the same type as original."""
    n_members = 3
    model = ensemble(torch_model_small_2d_2d, n_members=n_members)

    for member in model:
        assert isinstance(member, type(torch_model_small_2d_2d))


def test_ensemble_single_member(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Test ensemble with n_members=1."""
    model = ensemble(torch_model_small_2d_2d, n_members=1)

    assert isinstance(model, nn.ModuleList)
    assert len(model) == 1

def test_ensemble_multiple_members(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Test ensemble with n_members=5."""
    model = ensemble(torch_model_small_2d_2d, n_members=5)
    assert isinstance(model, nn.ModuleList)
    assert len(model) == 5


def test_reset_copy_returns_module(torch_model_small_2d_2d: nn.Sequential) -> None:
    """Test that _reset_copy returns a Module."""
    copied = _reset_copy(torch_model_small_2d_2d)

    assert isinstance(copied, nn.Module)
