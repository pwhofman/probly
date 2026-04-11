"""Fixtures for Sample representations."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.representation.sample.torch import TorchSample


@pytest.fixture
def torch_tensor_sample_2d() -> TorchSample:
    sample_array = torch.arange(12).reshape((3, 4))
    sample = TorchSample(sample_array, sample_dim=1)

    return sample
