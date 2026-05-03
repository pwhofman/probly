"""Tests for torch sparse log categorical distributions."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.distribution.torch_sparse_log_categorical import TorchSparseLogCategoricalDistribution


def test_sparse_log_categorical_validates_fields() -> None:
    with pytest.raises(TypeError, match="integer"):
        TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([0.0, 1.0]),
            logits=torch.tensor([-0.1, -0.2]),
        )

    with pytest.raises(TypeError, match="floating point"):
        TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([0, 1]),
            logits=torch.tensor([-1, -2]),
        )

    with pytest.raises(ValueError, match="identical shapes"):
        TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([0, 1]),
            logits=torch.tensor([[-0.1, -0.2]]),
        )

    with pytest.raises(ValueError, match="non-negative"):
        TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([0, -1]),
            logits=torch.tensor([-0.1, -0.2]),
        )


def test_sparse_log_categorical_indexes_and_moves_like_torch_representation() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([[0, 1], [1, 2]]),
        logits=torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
    )

    indexed = distribution[0]

    assert isinstance(indexed, TorchSparseLogCategoricalDistribution)
    assert torch.equal(indexed.group_ids, torch.tensor([0, 1]))
    assert torch.equal(indexed.logits, torch.tensor([-0.1, -0.2]))
    assert distribution.to(device=torch.device("cpu")) is distribution


def test_sparse_log_categorical_converts_to_dense_categorical_distribution() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([[0, 2, 2], [1, 3, 1]]),
        logits=torch.log(torch.tensor([[0.2, 0.3, 0.5], [0.25, 0.5, 0.25]])),
    )

    dense = distribution.to_categorical_distribution()

    assert isinstance(dense, TorchCategoricalDistribution)
    assert dense.num_classes == 4
    assert torch.allclose(
        dense.probabilities,
        torch.tensor([[0.2, 0.0, 0.8, 0.0], [0.0, 0.5, 0.0, 0.5]]),
    )


def test_sparse_log_categorical_allows_explicit_extra_dense_classes() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([0, 2]),
        logits=torch.log(torch.tensor([0.4, 0.6])),
    )

    dense = distribution.to_categorical_distribution(num_classes=5)

    assert dense.num_classes == 5
    assert torch.allclose(dense.probabilities, torch.tensor([0.4, 0.0, 0.6, 0.0, 0.0]))


def test_sparse_log_categorical_uniform_logits_reuses_groups() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([0, 0, 1]),
        logits=torch.tensor([-10.0, -20.0, 5.0]),
    )

    uniform = distribution.uniform_logits()

    assert uniform is not distribution
    assert uniform.group_ids is distribution.group_ids
    assert torch.equal(uniform.logits, torch.zeros_like(distribution.logits))
    assert torch.allclose(uniform.probabilities, torch.tensor([2 / 3, 1 / 3]))


def test_sparse_log_categorical_rejects_too_few_dense_classes() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([0, 2]),
        logits=torch.log(torch.tensor([0.4, 0.6])),
    )

    with pytest.raises(ValueError, match="maximum group id"):
        distribution.to_categorical_distribution(num_classes=2)
