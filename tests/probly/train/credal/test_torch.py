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


class TestCvarCeLoss:
    """Batch-wise top-delta cross-entropy used by credal DRO training."""

    def test_delta_one_equals_mean_ce(self) -> None:
        torch, _ = _torch_nn()
        from torch.nn import functional as F  # noqa: PLC0415

        from probly.train.credal.torch import cvar_ce_loss  # noqa: PLC0415

        torch.manual_seed(0)
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        loss = cvar_ce_loss(logits, targets, delta=1.0)
        assert torch.allclose(loss, F.cross_entropy(logits, targets))

    def test_keeps_only_highest_loss_samples(self) -> None:
        torch, _ = _torch_nn()
        from torch.nn import functional as F  # noqa: PLC0415

        from probly.train.credal.torch import cvar_ce_loss  # noqa: PLC0415

        # Samples 0 and 1 are confidently wrong, samples 2 and 3 confidently right.
        logits = torch.tensor(
            [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0]],
        )
        targets = torch.tensor([1, 1, 0, 0])
        per_sample = F.cross_entropy(logits, targets, reduction="none")
        loss = cvar_ce_loss(logits, targets, delta=0.5)
        assert torch.allclose(loss, per_sample[:2].mean())

    def test_gradient_only_flows_to_selected_samples(self) -> None:
        torch, _ = _torch_nn()
        from probly.train.credal.torch import cvar_ce_loss  # noqa: PLC0415

        logits = torch.tensor(
            [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0]],
            requires_grad=True,
        )
        targets = torch.tensor([1, 1, 0, 0])
        cvar_ce_loss(logits, targets, delta=0.5).backward()
        assert logits.grad is not None
        assert logits.grad[:2].abs().sum() > 0.0
        assert torch.allclose(logits.grad[2:], torch.zeros(2, 2))

    def test_tiny_batch_keeps_at_least_one_sample(self) -> None:
        torch, _ = _torch_nn()
        from torch.nn import functional as F  # noqa: PLC0415

        from probly.train.credal.torch import cvar_ce_loss  # noqa: PLC0415

        # floor(0.4 * 2) = 0 is clamped to one sample: the highest-loss one.
        logits = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
        targets = torch.tensor([1, 0])
        per_sample = F.cross_entropy(logits, targets, reduction="none")
        loss = cvar_ce_loss(logits, targets, delta=0.4)
        assert torch.allclose(loss, per_sample.max())

    def test_delta_out_of_range_raises(self) -> None:
        torch, _ = _torch_nn()
        from probly.train.credal.torch import cvar_ce_loss  # noqa: PLC0415

        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        with pytest.raises(ValueError, match="delta"):
            cvar_ce_loss(logits, targets, delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            cvar_ce_loss(logits, targets, delta=1.5)
