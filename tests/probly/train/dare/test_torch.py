"""Tests for the torch backend of probly.train.dare."""

from __future__ import annotations

import pytest


def _torch_nn():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestDareTorch:
    """DARE anti-regularizer behaves correctly above and below the threshold."""

    def test_anti_regularizer_zero_above_threshold(self) -> None:
        torch, nn = _torch_nn()
        from probly.train.dare.torch import dare_regularizer  # noqa: PLC0415

        model = nn.Linear(4, 3)
        loss = torch.tensor(2.0)
        threshold = torch.tensor(1.0)
        result = dare_regularizer(model, device="cpu", loss=loss, threshold=threshold)
        assert result.item() == 0.0

    def test_anti_regularizer_active_below_threshold(self) -> None:
        torch, nn = _torch_nn()
        from probly.train.dare.torch import dare_regularizer  # noqa: PLC0415

        model = nn.Linear(4, 3)
        loss = torch.tensor(0.5)
        threshold = torch.tensor(1.0)
        result = dare_regularizer(model, device="cpu", loss=loss, threshold=threshold)
        # Loss <= threshold -> non-zero anti-reg.
        assert torch.isfinite(result)

    def test_anti_regularizer_threshold_as_float(self) -> None:
        torch, nn = _torch_nn()
        from probly.train.dare.torch import dare_regularizer  # noqa: PLC0415

        model = nn.Linear(4, 3)
        loss = torch.tensor(0.5)
        result = dare_regularizer(model, device="cpu", loss=loss, threshold=1.0)
        assert torch.isfinite(result)

    def test_anti_regularizer_value_is_mean_log_squared_weight(self) -> None:
        torch, nn = _torch_nn()
        from probly.train.dare.torch import EPS_LOG, dare_regularizer  # noqa: PLC0415

        model = nn.Linear(4, 3)
        params = torch.cat([p.detach().flatten() for p in model.parameters()])
        expected = torch.log(params.pow(2) + EPS_LOG).mean()
        result = dare_regularizer(model, device="cpu", loss=torch.tensor(0.5), threshold=1.0)
        assert torch.allclose(result, expected)

    def test_anti_regularizer_gradient_zero_above_threshold(self) -> None:
        torch, nn = _torch_nn()
        from probly.train.dare.torch import dare_regularizer  # noqa: PLC0415

        model = nn.Linear(4, 3)
        dare_regularizer(model, device="cpu", loss=torch.tensor(2.0), threshold=1.0).backward()
        for param in model.parameters():
            assert param.grad is not None
            assert torch.count_nonzero(param.grad) == 0

    def test_anti_regularizer_gradient_nonzero_below_threshold(self) -> None:
        torch, nn = _torch_nn()
        from probly.train.dare.torch import dare_regularizer  # noqa: PLC0415

        model = nn.Linear(4, 3)
        dare_regularizer(model, device="cpu", loss=torch.tensor(0.5), threshold=1.0).backward()
        assert all(param.grad is not None and torch.count_nonzero(param.grad) > 0 for param in model.parameters())
