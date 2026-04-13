"""Tests for torch-backed categorical credal sets."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
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
