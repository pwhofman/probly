"""Tests for torch-backed categorical credal sets."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.credal_set._common import (
    create_distance_based_credal_set_from_center_and_radius,
)
from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample


def test_torch_convex_credal_set_from_distribution_sample() -> None:
    probs = torch.tensor(
        [
            [[0.1, 0.9], [0.4, 0.6]],
            [[0.2, 0.8], [0.5, 0.5]],
            [[0.15, 0.85], [0.45, 0.55]],
        ],
        dtype=torch.float64,
    )
    sample = TorchSample(
        tensor=TorchCategoricalDistribution(probs),
        sample_dim=0,
    )

    cset = TorchConvexCredalSet.from_torch_sample(sample)

    assert isinstance(cset.tensor, TorchCategoricalDistribution)
    assert tuple(cset.tensor.probabilities.shape) == (2, 3, 2)


def test_torch_convex_credal_set_barycenter_averages_normalized_probabilities() -> None:
    vertices = torch.tensor([[1.0, 1.0], [9.0, 1.0]], dtype=torch.float64)
    cset = TorchConvexCredalSet(tensor=TorchCategoricalDistribution(vertices))

    barycenter = cset.barycenter

    assert isinstance(barycenter, TorchCategoricalDistribution)
    assert torch.allclose(barycenter.probabilities, torch.tensor([0.7, 0.3], dtype=torch.float64))


def test_torch_probability_intervals_numpy_and_shape_ops() -> None:
    probs = torch.tensor(
        [
            [[0.2, 0.8], [0.6, 0.4]],
            [[0.1, 0.9], [0.5, 0.5]],
        ],
        dtype=torch.float64,
    )
    sample = TorchSample(
        tensor=TorchCategoricalDistribution(probs),
        sample_dim=0,
    )

    cset = TorchProbabilityIntervalsCredalSet.from_torch_sample(sample)
    arr = np.asarray(cset)

    assert arr.shape == (2, 2, 2)

    expanded = torch.unsqueeze(cset, dim=0)
    assert isinstance(expanded, TorchProbabilityIntervalsCredalSet)
    assert tuple(expanded.lower_bounds.shape) == (1, 2, 2)
    assert tuple(expanded.upper_bounds.shape) == (1, 2, 2)


def test_distance_credal_set_from_categorical_distribution() -> None:
    """Factory should accept TorchCategoricalDistribution directly."""
    probs = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], dtype=torch.float64)
    center = TorchCategoricalDistribution(probs)
    radius = torch.tensor(0.1, dtype=torch.float64)

    result = create_distance_based_credal_set_from_center_and_radius(center, radius)

    assert isinstance(result, TorchDistanceBasedCredalSet)
    assert result.nominal is center  # should reuse, not re-wrap
    assert torch.equal(result.radius, radius)
