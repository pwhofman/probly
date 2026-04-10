"""Tests for the vector scaling implementation with torch."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.calibration.scaling.torch_vector import TorchVector


def test_forward(torch_setup_multiclass: nn.Module) -> None:
    base, inputs, _ = torch_setup_multiclass

    vector_model = TorchVector(base, num_classes=3)

    vector_model.w.values = [0.5, 0.25, 0.1]
    vector_model.b.values = [1, 2, 3]

    logits_base = base(inputs)
    logits_expected = logits_base * vector_model.w + vector_model.b
    logits_scaled = vector_model.forward(inputs)

    assert vector_model.w.shape == (3,)
    assert vector_model.b.shape == (3,)
    assert torch.allclose(logits_scaled, logits_expected, atol=1e-5)


def test_fit(torch_setup_multiclass: nn.Module) -> None:
    base, inputs, labels = torch_setup_multiclass

    vector_model = TorchVector(base, num_classes=3)

    dataloader = DataLoader(TensorDataset(inputs, labels), batch_size=10)
    w_unoptimized = vector_model.w.detach().clone()
    b_unoptimized = vector_model.b.detach().clone()

    vector_model.fit(dataloader, learning_rate=0.01, max_iter=50)

    assert vector_model.w.values != w_unoptimized
    assert vector_model.b.values != b_unoptimized


def test_predict(torch_setup_multiclass: nn.Module) -> None:
    base, inputs, _ = torch_setup_multiclass
    vector_model = TorchVector(base, num_classes=3)

    predictions = vector_model.predict(inputs)

    assert predictions.shape == (20, 3)
    assert torch.all(predictions >= 0)
    assert torch.all(predictions <= 1)
