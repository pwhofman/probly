"""Tests for ``probly.method.credal_net``."""

from __future__ import annotations

import pytest


def _torch_nn():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestCredalNetMethod:
    """``credal_net`` is a wrapper over ``interval_classifier``."""

    def test_credal_net_transforms_classifier(self) -> None:
        torch, nn = _torch_nn()
        from probly.method.credal_net import credal_net  # noqa: PLC0415
        from probly.predictor import LogitClassifier  # noqa: PLC0415

        base = nn.Sequential(nn.Linear(4, 3))
        net = credal_net(base, predictor_type=LogitClassifier)
        x = torch.randn(2, 4)
        from probly.predictor import predict  # noqa: PLC0415

        result = predict(net, x)
        assert result is not None
