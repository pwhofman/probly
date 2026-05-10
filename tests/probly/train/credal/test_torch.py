"""Tests for the torch backend of probly.train.credal."""

from __future__ import annotations

import pytest


def _torch_nn():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestCredalTrainTorch:
    """Cross-entropy on intersection probability of an interval-valued prediction."""

    def test_collapsed_interval_loss_finite(self) -> None:
        torch, _ = _torch_nn()
        from probly.train.credal.torch import intersection_probability_ce_loss  # noqa: PLC0415

        # When lower == upper, intersection probability matches the point estimate.
        probs = torch.tensor([[0.7, 0.2, 0.1]])
        packed = torch.cat([probs, probs], dim=-1)
        targets = torch.tensor([0])
        loss = intersection_probability_ce_loss(packed, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_loss_decreases_with_more_confident_prediction(self) -> None:
        torch, _ = _torch_nn()
        from probly.train.credal.torch import intersection_probability_ce_loss  # noqa: PLC0415

        confident = torch.tensor([[0.9, 0.05, 0.05]])
        unconfident = torch.tensor([[0.34, 0.33, 0.33]])
        packed_conf = torch.cat([confident, confident], dim=-1)
        packed_unconf = torch.cat([unconfident, unconfident], dim=-1)
        targets = torch.tensor([0])
        loss_conf = intersection_probability_ce_loss(packed_conf, targets)
        loss_unconf = intersection_probability_ce_loss(packed_unconf, targets)
        assert loss_conf.item() < loss_unconf.item()
