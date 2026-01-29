from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from probly.data_generation.pytorch_generator import PyTorchDataGenerator


@pytest.fixture
def mock_dataset():
    # A dataset with 4 samples, each with 2 features
    x = torch.randn(4, 2)
    y = torch.tensor([0, 1, 0, 1])
    dataset = MagicMock()
    dataset.__len__.return_value = 4
    # Mocking the dataloader behavior: returns (batch_x, batch_y)
    dataset.__getitem__ = MagicMock(side_effect=[(x[i], y[i]) for i in range(4)])
    return dataset, x, y


@pytest.fixture
def mock_model():
    model = MagicMock(spec=torch.nn.Module)
    # Mock model(x) to return simple logits
    # Predicts class 0 for even indicies, class 1 for odd
    model.side_effect = lambda x: torch.tensor([[10.0, 0.0] if i % 2 == 0 else [0.0, 10.0] for i in range(len(x))])
    return model


def test_count_method(mock_model, mock_dataset):
    ds, _, _ = mock_dataset
    gen = PyTorchDataGenerator(mock_model, ds)

    # Create a test tensor
    test_tensor = torch.tensor([0, 1, 1, 2, 2, 2])
    counts = gen._count(test_tensor)  # noqa: SLF001

    assert counts == {0: 1, 1: 2, 2: 3}
    assert isinstance(counts[0], int)


def test_initialization(mock_model, mock_dataset):
    ds, _, _ = mock_dataset
    gen = PyTorchDataGenerator(mock_model, ds, batch_size=2)

    assert gen.batch_size == 2
    mock_model.eval.assert_called_once()
    assert gen.device in ["cpu", "cuda"]


def test_generate_logic(mock_model, mock_dataset):
    ds, _, _ = mock_dataset
    gen = PyTorchDataGenerator(mock_model, ds, batch_size=2)
    results = gen.generate()

    assert results["metrics"]["accuracy"] == 1.0
    assert results["info"]["dataset_size"] == 4
    assert results["class_distribution"]["ground_truth"][0] == 2


def test_save_load_cycle(mock_model, mock_dataset, tmp_path):
    ds, _, _ = mock_dataset
    gen = PyTorchDataGenerator(mock_model, ds)
    gen.generate()

    file_path = tmp_path / "results.json"
    gen.save(str(file_path))

    # Create a new generator and load the data
    new_gen = PyTorchDataGenerator(mock_model, ds)
    loaded_data = new_gen.load(str(file_path))

    assert loaded_data["metrics"]["accuracy"] == gen.results["metrics"]["accuracy"]
    assert loaded_data["info"]["framework"] == "pytorch"
