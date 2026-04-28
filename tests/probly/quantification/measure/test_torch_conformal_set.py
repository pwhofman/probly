"""Tests for torch conformal-set uncertainty measures."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification import decompose, measure
from probly.quantification.decomposition.decomposition import ConstantTotalDecomposition
from probly.representation.conformal_set.torch import TorchIntervalConformalSet, TorchOneHotConformalSet


def test_measure_uses_set_size_for_torch_one_hot_conformal_set() -> None:
    conformal_set = TorchOneHotConformalSet(
        torch.tensor(
            [
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=torch.int64,
        )
    )

    uncertainty = measure(conformal_set)

    assert torch.equal(uncertainty, conformal_set.set_size)


def test_decompose_wraps_torch_conformal_set_size_as_total_uncertainty() -> None:
    lower = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    upper = torch.tensor([[0.5, 2.5], [5.0, 4.0]], dtype=torch.float64)
    conformal_set = TorchIntervalConformalSet.from_tensor_samples(lower, upper)

    decomposition = decompose(conformal_set)

    assert isinstance(decomposition, ConstantTotalDecomposition)
    assert torch.allclose(decomposition.total, conformal_set.set_size, rtol=1e-12, atol=1e-12)
