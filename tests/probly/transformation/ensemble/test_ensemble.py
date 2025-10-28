import pytest
import torch
import torch.nn as nn
from probly.predictor import Predictor
from probly.transformation.ensemble import ensemble

@pytest.fixture
def dummy_model() -> nn.Module:
    """Ein simples Torch-Modul für Tests."""

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 2)

        def forward(self, x):
            return self.linear(x)

    return DummyNet()

@pytest.fixture
def sample_data():
    return torch.randn(3, 5)

def test_ensemble_with_zero_members_returns_empty_list(dummy_model: nn.Module):
    invalid_size = 0
    transformed_model = ensemble(dummy_model, n_members=invalid_size)

    assert isinstance(transformed_model, nn.ModuleList)
    assert len(transformed_model) == 0

def test_ensemble_transformation_returns_predictor(dummy_model: nn.Module):
    N_MEMBERS = 4
    transformed_model = ensemble(dummy_model, n_members=N_MEMBERS)

    assert isinstance(transformed_model, Predictor)
    assert transformed_model is not dummy_model

def test_ensemble_returns_modulelist_of_correct_size(dummy_model: nn.Module, sample_data: torch.Tensor):
    N_MEMBERS = 5
    transformed_model = ensemble(dummy_model, n_members=N_MEMBERS)

    assert isinstance(transformed_model, nn.ModuleList)
    assert len(transformed_model) == N_MEMBERS
    assert isinstance(transformed_model[0], type(dummy_model))
