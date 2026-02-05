"""Conftest for isotonic regression tests containing necessary fixtures."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

SetupReturnType = tuple[nn.Module, nn.Module, Tensor, DataLoader, DataLoader]


@pytest.fixture
def binary_torch_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(10, 2),
    )


@pytest.fixture
def multiclass_torch_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(10, 4),
    )


@pytest.fixture
def setup(binary_torch_model: nn.Sequential, multiclass_torch_model: nn.Sequential) -> SetupReturnType:
    device = torch.device("cpu")
    base_model_multiclass = multiclass_torch_model.to(device)
    base_model_binary = binary_torch_model.to(device)

    inputs = torch.randn(20, 10)
    labels_multiclass = torch.randint(0, 3, (20,))
    labels_binary = torch.randint(0, 2, (20,))

    loader_multiclass = DataLoader(TensorDataset(inputs, labels_multiclass), batch_size=10)
    loader_binary = DataLoader(TensorDataset(inputs, labels_binary), batch_size=10)

    return base_model_multiclass, base_model_binary, inputs, loader_multiclass, loader_binary
