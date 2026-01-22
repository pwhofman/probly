"""Fixtures for Sample representations."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.representation.sampling.torch_sample import TorchTensorSample


@pytest.fixture
def torch_tensor_sample_2d() -> TorchTensorSample:
    sample_array = torch.arange(12).reshape((3, 4))
    sample = TorchTensorSample(sample_array, sample_dim=1)

    return sample
