"""Tests for Dirichlet relative likelihood non-conformity score."""

from __future__ import annotations

import numpy as np
import pytest

from probly.conformal_scores.dirichlet_relative_likelihood._common import dirichlet_rl_score_func


def test_score_confident_correct_class() -> None:
    """When the true class has the highest alpha, score should be 0."""
    alphas = np.array([[10.0, 1.0, 1.0]])
    y_true = np.array([0])
    scores = dirichlet_rl_score_func(alphas, y_true)
    assert scores[0] == pytest.approx(0.0)


def test_score_wrong_class_high() -> None:
    """When the true class has low alpha, score should be near 1."""
    alphas = np.array([[1.0, 10.0, 1.0]])
    y_true = np.array([0])
    scores = dirichlet_rl_score_func(alphas, y_true)
    assert scores[0] == pytest.approx(1.0 - 1.0 / 10.0)


def test_score_uniform_is_zero() -> None:
    """When all alphas are equal, score should be 0 regardless of class."""
    alphas = np.array([[5.0, 5.0, 5.0]])
    y_true = np.array([1])
    scores = dirichlet_rl_score_func(alphas, y_true)
    assert scores[0] == pytest.approx(0.0)


def test_score_batch() -> None:
    """Test batch computation."""
    alphas = np.array([[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [5.0, 5.0, 5.0]])
    y_true = np.array([0, 0, 2])
    scores = dirichlet_rl_score_func(alphas, y_true)
    np.testing.assert_allclose(scores, [0.0, 0.9, 0.0])


class TestTorch:
    """Torch-specific tests."""

    @pytest.fixture(autouse=True)
    def _skip_no_torch(self) -> None:
        pytest.importorskip("torch")

    def test_torch_matches_numpy(self) -> None:
        import torch  # noqa: PLC0415

        alphas_np = np.array([[10.0, 2.0, 1.0], [1.0, 5.0, 3.0]])
        y_true_np = np.array([0, 2])
        scores_np = dirichlet_rl_score_func(alphas_np, y_true_np)

        alphas_t = torch.tensor(alphas_np)
        y_true_t = torch.tensor(y_true_np)
        scores_t = dirichlet_rl_score_func(alphas_t, y_true_t)
        np.testing.assert_allclose(scores_t.numpy(), scores_np, atol=1e-7)

    def test_torch_dirichlet_distribution_input(self) -> None:
        import torch  # noqa: PLC0415

        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        alphas = torch.tensor([[10.0, 2.0, 1.0], [1.0, 5.0, 3.0]])
        dirichlet = TorchDirichletDistribution(alphas)
        y_true = torch.tensor([0, 2])
        scores = dirichlet_rl_score_func(dirichlet, y_true)
        expected = 1.0 - torch.tensor([10.0 / 10.0, 3.0 / 5.0])
        assert torch.allclose(scores, expected)
