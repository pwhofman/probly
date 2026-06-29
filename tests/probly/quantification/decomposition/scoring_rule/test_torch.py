"""Tests for the scoring-rule decomposition on PyTorch representations."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.quantification import (
    BrierLoss,
    LogLoss,
    SecondOrderEntropyDecomposition,
    SecondOrderScoringRuleDecomposition,
    SecondOrderZeroOneDecomposition,
    SphericalLoss,
    ZeroOneLoss,
)
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchProbabilityCategoricalDistribution,
)

_PROBS = torch.tensor(
    [
        [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
        [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
        [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
    ],
    dtype=torch.float64,
)


def _sample() -> TorchCategoricalDistributionSample:
    return TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(_PROBS),
        sample_dim=0,
    )


def test_torch_log_loss_matches_entropy_decomposition() -> None:
    sr = SecondOrderScoringRuleDecomposition(_sample(), LogLoss())
    ent = SecondOrderEntropyDecomposition(_sample())
    assert torch.allclose(sr.total, ent.total, rtol=1e-10, atol=1e-12)
    assert torch.allclose(sr.aleatoric, ent.aleatoric, rtol=1e-10, atol=1e-12)
    assert torch.allclose(sr.epistemic, ent.epistemic, rtol=1e-10, atol=1e-12)


def test_torch_zero_one_loss_matches_zero_one_decomposition() -> None:
    sr = SecondOrderScoringRuleDecomposition(_sample(), ZeroOneLoss())
    zo = SecondOrderZeroOneDecomposition(_sample())
    assert torch.allclose(sr.total, zo.total, rtol=1e-12, atol=1e-12)
    assert torch.allclose(sr.aleatoric, zo.aleatoric, rtol=1e-12, atol=1e-12)
    assert torch.allclose(sr.epistemic, zo.epistemic, rtol=1e-12, atol=1e-12)


def test_torch_brier_loss_matches_gini_closed_form() -> None:
    mean = _PROBS.mean(dim=0)
    tu = 1.0 - torch.sum(mean**2, dim=-1)
    au = torch.mean(1.0 - torch.sum(_PROBS**2, dim=-1), dim=0)
    sr = SecondOrderScoringRuleDecomposition(_sample(), BrierLoss())
    assert torch.allclose(sr.total, tu, rtol=1e-12, atol=1e-12)
    assert torch.allclose(sr.aleatoric, au, rtol=1e-12, atol=1e-12)


def test_torch_spherical_loss_is_additive_and_nonnegative_eu() -> None:
    sr = SecondOrderScoringRuleDecomposition(_sample(), SphericalLoss())
    assert torch.allclose(sr.total, sr.aleatoric + sr.epistemic, rtol=1e-12, atol=1e-12)
    assert torch.all(sr.epistemic >= -1e-12)


def test_torch_notion_indexing_matches_properties() -> None:
    sr = SecondOrderScoringRuleDecomposition(_sample(), BrierLoss())
    assert torch.allclose(sr["tu"], sr.total)
    assert torch.allclose(sr["au"], sr.aleatoric)
    assert torch.allclose(sr["eu"], sr.epistemic)
