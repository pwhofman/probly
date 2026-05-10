"""Tests for the torch backend of dirichlet_softplus_activation."""

from __future__ import annotations

import pytest


def _torch_nn():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestDirichletSoftplusActivationTorch:
    """Wrap a Torch model so it emits Dirichlet alpha = softplus(logits) + 1."""

    def test_appends_softplus_and_plus_one(self) -> None:
        _, nn = _torch_nn()
        from probly.transformation.dirichlet_softplus_activation.torch import (  # noqa: PLC0415
            _AddOne,
            append_activation_torch,
        )

        base = nn.Linear(4, 3)
        wrapped = append_activation_torch(base)
        assert isinstance(wrapped, nn.Sequential)
        assert len(wrapped) == 3
        assert wrapped[0] is base
        assert isinstance(wrapped[1], nn.Softplus)
        assert isinstance(wrapped[2], _AddOne)

    def test_alpha_is_strictly_greater_than_one(self) -> None:
        torch, nn = _torch_nn()
        from probly.transformation.dirichlet_softplus_activation.torch import append_activation_torch  # noqa: PLC0415

        base = nn.Linear(4, 3)
        wrapped = append_activation_torch(base)
        x = torch.randn(2, 4)
        out = wrapped(x)
        assert torch.all(out > 1.0)

    def test_add_one_module_forward(self) -> None:
        torch, _ = _torch_nn()
        from probly.transformation.dirichlet_softplus_activation.torch import _AddOne  # noqa: PLC0415

        m = _AddOne()
        x = torch.tensor([[0.0, 1.0, -2.0]])
        out = m(x)
        torch.testing.assert_close(out, x + 1.0)

    def test_dirichlet_softplus_high_level_call(self) -> None:
        """Top-level dirichlet_softplus_activation wraps a base model."""
        torch, nn = _torch_nn()
        from probly.transformation.dirichlet_softplus_activation import (  # noqa: PLC0415
            dirichlet_softplus_activation,
        )

        base = nn.Linear(4, 3)
        wrapped = dirichlet_softplus_activation(base)
        x = torch.randn(2, 4)
        alpha = wrapped(x)
        assert torch.all(alpha > 1.0)


def _torch_modules():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestAddOneInDirichletSoftplus:
    """`_AddOne` is reused conceptually in DEUP — sanity-check it remains correct."""

    def test_add_one_module(self) -> None:
        torch, _ = _torch_modules()
        from probly.transformation.dirichlet_softplus_activation.torch import _AddOne  # noqa: PLC0415

        m = _AddOne()
        x = torch.tensor([1.0, 2.0])
        out = m(x)
        torch.testing.assert_close(out, torch.tensor([2.0, 3.0]))
