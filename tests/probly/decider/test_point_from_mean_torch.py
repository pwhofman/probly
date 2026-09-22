"""Torch tests for the point_from_mean decider."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from probly.decider import point_from_mean
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution
from probly.representation.sample.torch import TorchSample


def test_point_from_mean_passes_tensors_through() -> None:
    prediction = torch.tensor([1.0, 2.0])

    assert point_from_mean(prediction) is prediction


def test_point_from_mean_reduces_torch_gaussian_to_its_mean() -> None:
    mean = torch.tensor([0.5, -1.0])
    gaussian = TorchGaussianDistribution(mean=mean, var=torch.tensor([1.0, 2.0]))

    assert torch.equal(point_from_mean(gaussian), mean)


def test_point_from_mean_reduces_torch_sample_to_sample_mean() -> None:
    sample = TorchSample(tensor=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), sample_dim=0)

    assert torch.allclose(point_from_mean(sample), torch.tensor([3.0, 4.0]))
