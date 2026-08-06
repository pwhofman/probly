"""PyTorch backend tests for selective prediction."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from probly.evaluation.selective_prediction import selective_prediction  # noqa: E402


def test_selective_prediction_exact_values() -> None:
    criterion = torch.tensor([3.0, 1.0, 2.0, 0.0])
    losses = torch.tensor([30.0, 10.0, 20.0, 0.0])
    aurc, bin_losses = selective_prediction(criterion, losses, n_bins=2)
    assert isinstance(aurc, torch.Tensor)
    assert isinstance(bin_losses, torch.Tensor)
    torch.testing.assert_close(bin_losses, torch.tensor([15.0, 5.0]))
    torch.testing.assert_close(aurc, torch.tensor(10.0))


def test_selective_prediction_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    criterion = rng.random(100)
    losses = rng.random(100)
    aurc_np, bins_np = selective_prediction(criterion, losses, n_bins=10)
    aurc_torch, bins_torch = selective_prediction(torch.from_numpy(criterion), torch.from_numpy(losses), n_bins=10)
    np.testing.assert_allclose(bins_torch.numpy(), bins_np)
    np.testing.assert_allclose(float(aurc_torch), aurc_np)


def test_selective_prediction_too_many_bins() -> None:
    with pytest.raises(ValueError, match="The number of bins can not be larger than the number of elements criterion"):
        selective_prediction(torch.rand(5), torch.rand(5), n_bins=10)
