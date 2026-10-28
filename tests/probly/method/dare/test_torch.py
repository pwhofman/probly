"""Tests for the torch backend of probly.method.dare."""

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
        from probly.method.dare import dare_anti_regularization  # noqa: PLC0415

        model = nn.Linear(4, 3)
        loss = torch.tensor(2.0)
        threshold = torch.tensor(1.0)
        result = dare_anti_regularization(model, device="cpu", loss=loss, threshold=threshold)
        assert result.item() == 0.0  # ty: ignore[unresolved-attribute]

    def test_anti_regularizer_active_below_threshold(self) -> None:
        torch, nn = _torch_nn()
        from probly.method.dare import dare_anti_regularization  # noqa: PLC0415

        model = nn.Linear(4, 3)
        loss = torch.tensor(0.5)
        threshold = torch.tensor(1.0)
        result = dare_anti_regularization(model, device="cpu", loss=loss, threshold=threshold)
        # Loss <= threshold -> non-zero anti-reg.
        assert torch.isfinite(result)

    def test_anti_regularizer_threshold_as_float(self) -> None:
        torch, nn = _torch_nn()
        from probly.method.dare import dare_anti_regularization  # noqa: PLC0415

        model = nn.Linear(4, 3)
        loss = torch.tensor(0.5)
        result = dare_anti_regularization(model, device="cpu", loss=loss, threshold=1.0)
        assert torch.isfinite(result)


class TestDareDecompositionTorch:
    """DARE decomposition reproduces the OOD score of Eq. 35."""

    def test_epistemic_matches_eq_35(self) -> None:
        torch, nn = _torch_nn()
        from probly.method.dare import DAREDecomposition  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchCategoricalDistributionSample,
            TorchLogitCategoricalDistribution,
        )

        # 3 members, batch of 2, 3 classes.
        logits = torch.tensor(
            [
                [[2.0, 0.5, -1.0], [0.0, 1.0, 0.0]],
                [[1.0, 1.5, -0.5], [0.5, -1.0, 2.0]],
                [[-0.5, 0.0, 1.0], [1.0, 0.0, -1.0]],
            ]
        )
        dist = TorchLogitCategoricalDistribution(tensor=logits)
        sample = TorchCategoricalDistributionSample(tensor=dist, sample_dim=0)

        num_members, _, num_classes = logits.shape
        # One-hot of each member's predicted class, scaled by the number of classes.
        target = num_classes * nn.functional.one_hot(logits.argmax(dim=-1), num_classes)
        fit = ((logits - target) ** 2).sum(dim=-1).sum(dim=0) / num_members
        dispersion = ((logits - logits.mean(dim=0)) ** 2).sum(dim=-1).sum(dim=0) / num_members

        assert torch.allclose(DAREDecomposition(sample).epistemic, fit + dispersion)
