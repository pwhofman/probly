"""Tests for PyTorch RAPS implementation."""

from __future__ import annotations

import numpy as np
import torch

from probly.conformal_prediction.scores.raps.common import raps_score_func
from probly.conformal_prediction.scores.raps.torch import raps_score_torch


def test_raps_score_torch_basic() -> None:
    """Test raps_score_torch with basic data."""
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
    )

    scores = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (2, 3)
    assert torch.all(scores >= 0)
    # RAPS adds regularization, so can exceed 1
    assert scores[0, 0] > 0


def test_raps_score_torch_consistency_with_numpy() -> None:
    """Test that Torch implementation produces similar results to NumPy."""
    # Create test data
    np_probs = np.array(
        [
            [0.4, 0.3, 0.3],
            [0.6, 0.2, 0.2],
            [0.1, 0.8, 0.1],
        ],
    )
    torch_probs = torch.tensor(np_probs)

    # Get scores from both implementations

    np_scores: np.ndarray = raps_score_func(np_probs, lambda_reg=0.1, k_reg=0)
    torch_scores = raps_score_torch(torch_probs, lambda_reg=0.1, k_reg=0)

    # Convert torch to numpy for comparison
    torch_scores_np = torch_scores.numpy()

    # Check shape and approximate equality
    assert np_scores.shape == torch_scores_np.shape
    assert np.allclose(np_scores, torch_scores_np, rtol=1e-5, atol=1e-5)


def test_raps_score_torch_different_lambda() -> None:
    """Test raps_score_torch with different regularization parameters."""
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
    )

    scores_lambda_01 = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)
    scores_lambda_10 = raps_score_torch(probs, lambda_reg=1.0, k_reg=0)

    # Higher lambda should produce higher scores due to larger penalty
    assert torch.all(scores_lambda_10 >= scores_lambda_01)
    assert scores_lambda_01.shape == (2, 3)
    assert scores_lambda_10.shape == (2, 3)


def test_raps_score_torch_different_k_reg() -> None:
    """Test raps_score_torch with different k_reg values."""
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
    )

    scores_k0 = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)
    scores_k1 = raps_score_torch(probs, lambda_reg=0.1, k_reg=1)
    scores_k2 = raps_score_torch(probs, lambda_reg=0.1, k_reg=2)

    # With higher k_reg, penalty starts later, so scores should be lower
    assert torch.all(scores_k1 <= scores_k0)  # k_reg=1 reduces penalty compared to k_reg=0
    assert torch.all(scores_k2 <= scores_k1)  # k_reg=2 further reduces penalty


def test_raps_score_torch_with_epsilon() -> None:
    """Test raps_score_torch with epsilon parameter."""
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
    )

    scores_no_eps = raps_score_torch(probs, lambda_reg=0.1, k_reg=0, epsilon=0.0)
    scores_with_eps = raps_score_torch(probs, lambda_reg=0.1, k_reg=0, epsilon=0.01)

    # With epsilon, all scores should be higher
    assert torch.all(scores_with_eps >= scores_no_eps)
    # Difference should be exactly epsilon
    diff = scores_with_eps - scores_no_eps
    assert torch.allclose(diff, torch.tensor(0.01), rtol=1e-5)


def test_raps_score_torch_gradient_computation() -> None:
    """Test that raps_score_torch supports gradient computation."""
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
        requires_grad=True,
    )

    scores = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)

    # Create a dummy loss
    loss = torch.sum(scores)

    # Compute gradients
    loss.backward()

    # Check gradients were computed
    assert probs.grad is not None
    assert probs.grad.shape == probs.shape
    # Gradients should not be all zeros
    assert not torch.allclose(probs.grad, torch.tensor(0.0), atol=1e-5)


def test_raps_score_torch_batch_independence() -> None:
    """Test that scores are computed independently per sample."""
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
            [0.3, 0.3, 0.4],
        ],
    )

    # Compute scores for all samples
    all_scores = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)

    # Compute scores for each sample individually
    individual_scores = []
    for i in range(probs.shape[0]):
        single_probs = probs[i : i + 1, :]  # Keep batch dimension
        single_scores = raps_score_torch(single_probs, lambda_reg=0.1, k_reg=0)
        individual_scores.append(single_scores[0])  # Remove batch dimension

    # Stack individual scores
    stacked_scores = torch.stack(individual_scores, dim=0)

    # Results should be identical
    assert torch.allclose(all_scores, stacked_scores, rtol=1e-5, atol=1e-5)


def test_raps_score_torch_device_agnostic() -> None:
    """Test that raps_score_torch works on different devices."""
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
    )

    # Test on CPU (always available)
    scores_cpu = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)
    assert scores_cpu.shape == (2, 3)
    assert torch.all(scores_cpu >= 0)

    # Test on GPU if available
    if torch.cuda.is_available():
        probs_gpu = probs.cuda()
        scores_gpu = raps_score_torch(probs_gpu, lambda_reg=0.1, k_reg=0)
        assert scores_gpu.shape == (2, 3)
        assert torch.all(scores_gpu >= 0)
        # Results should be the same (within tolerance)
        assert torch.allclose(scores_cpu, scores_gpu.cpu(), rtol=1e-5, atol=1e-5)


def test_raps_score_torch_edge_case_single_sample() -> None:
    """Test raps_score_torch with single sample."""
    probs = torch.tensor([[0.5, 0.3, 0.2]])
    scores = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (1, 3)
    assert torch.all(scores >= 0)


def test_raps_score_torch_edge_case_single_class() -> None:
    """Test raps_score_torch with single class."""
    probs = torch.tensor([[1.0]])
    scores = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (1, 1)
    # With one class, score should be prob + epsilon (penalty is 0 when k_reg=0)
    expected = 1.0 + 0.01  # prob + epsilon
    assert torch.allclose(scores[0, 0], torch.tensor(expected), rtol=1e-5)


def test_raps_score_torch_large_batch() -> None:
    """Test raps_score_torch with large batch size."""
    rng = np.random.default_rng(42)
    np_probs = rng.dirichlet([1, 1, 1], size=1000).astype(np.float32)
    probs = torch.tensor(np_probs)

    scores = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (1000, 3)
    assert torch.all(scores >= 0)


def test_raps_score_torch_dtype_preservation() -> None:
    """Test that raps_score_torch preserves input dtype."""
    # Test with float32
    probs_f32 = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
        dtype=torch.float32,
    )

    scores_f32 = raps_score_torch(probs_f32, lambda_reg=0.1, k_reg=0)
    assert scores_f32.dtype == torch.float32

    # Test with float64
    probs_f64 = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
        dtype=torch.float64,
    )

    scores_f64 = raps_score_torch(probs_f64, lambda_reg=0.1, k_reg=0)
    assert scores_f64.dtype == torch.float64


def test_raps_score_torch_monotonic_with_probability() -> None:
    """Test that scores are monotonic with respect to probability ordering."""
    probs = torch.tensor([[0.6, 0.25, 0.15]])
    scores = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)

    # Get indices sorted by probability (descending)
    _, sorted_indices = torch.sort(probs[0], descending=True)

    # In RAPS, cumulative sum should be non-decreasing
    # But highest probability class should have lowest score
    assert scores[0, 0] <= scores[0, 1]  # 0.6 vs 0.25
    assert scores[0, 0] <= scores[0, 2]  # 0.6 vs 0.15


def test_raps_score_torch_deterministic() -> None:
    """Test that raps_score_torch is deterministic."""
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.7, 0.2],
        ],
    )

    scores1 = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)
    scores2 = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)

    assert torch.allclose(scores1, scores2, rtol=1e-5, atol=1e-5)


def test_raps_score_torch_zeros_input() -> None:
    """Test raps_score_torch with edge case of zero probabilities."""
    probs = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # One class has all probability
            [0.0, 1.0, 0.0],
        ],
    )

    scores = raps_score_torch(probs, lambda_reg=0.1, k_reg=0)

    assert scores.shape == (2, 3)
    assert torch.all(scores >= 0)
    # Class with probability 1 should have score = 1 + epsilon
    assert torch.allclose(scores[0, 2], torch.tensor(1.01), rtol=1e-5)  # 1.0 + 0.01 epsilon
    assert torch.allclose(scores[1, 1], torch.tensor(1.01), rtol=1e-5)  # 1.0 + 0.01 epsilon


def test_raps_score_torch_extreme_probabilities() -> None:
    """Test raps_score_torch with extreme probability distributions."""
    # Very concentrated distribution
    probs_concentrated = torch.tensor([[0.99, 0.005, 0.005]])
    scores_concentrated = raps_score_torch(probs_concentrated, lambda_reg=0.1, k_reg=0)

    assert scores_concentrated.shape == (1, 3)
    assert torch.all(scores_concentrated >= 0)

    # Most probable class should have lowest score
    assert scores_concentrated[0, 0] <= scores_concentrated[0, 1]
    assert scores_concentrated[0, 0] <= scores_concentrated[0, 2]
