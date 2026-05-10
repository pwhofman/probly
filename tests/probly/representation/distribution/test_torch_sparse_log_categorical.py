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
            entry_logits=torch.tensor([-0.1, -0.2]),
        )

    with pytest.raises(TypeError, match="floating point"):
        TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([0, 1]),
            entry_logits=torch.tensor([-1, -2]),
        )

    with pytest.raises(ValueError, match="identical shapes"):
        TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([0, 1]),
            entry_logits=torch.tensor([[-0.1, -0.2]]),
        )

    with pytest.raises(ValueError, match="non-negative"):
        TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([0, -1]),
            entry_logits=torch.tensor([-0.1, -0.2]),
        )


def test_sparse_log_categorical_indexes_and_moves_like_torch_representation() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([[0, 1], [1, 2]]),
        entry_logits=torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
    )

    indexed = distribution[0]

    assert isinstance(indexed, TorchSparseLogCategoricalDistribution)
    assert torch.equal(indexed.group_ids, torch.tensor([0, 1]))
    assert torch.equal(indexed.entry_logits, torch.tensor([-0.1, -0.2]))
    assert distribution.to(device=torch.device("cpu")) is distribution


def test_sparse_log_categorical_converts_to_dense_categorical_distribution() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([[0, 2, 2], [1, 3, 1]]),
        entry_logits=torch.log(torch.tensor([[0.2, 0.3, 0.5], [0.25, 0.5, 0.25]])),
    )

    dense = distribution.to_dense()

    assert isinstance(dense, TorchCategoricalDistribution)
    assert dense.num_classes == 4
    assert torch.allclose(
        dense.probabilities,
        torch.tensor([[0.2, 0.0, 0.8, 0.0], [0.0, 0.5, 0.0, 0.5]]),
    )


def test_sparse_log_categorical_allows_explicit_extra_dense_classes() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([0, 2]),
        entry_logits=torch.log(torch.tensor([0.4, 0.6])),
    )

    dense = distribution.to_dense(num_classes=5)

    assert dense.num_classes == 5
    assert torch.allclose(dense.probabilities, torch.tensor([0.4, 0.0, 0.6, 0.0, 0.0]))


def test_sparse_log_categorical_uniform_logits_reuses_groups() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([0, 0, 1]),
        entry_logits=torch.tensor([-10.0, -20.0, 5.0]),
    )

    uniform = distribution.uniform_logits()

    assert uniform is not distribution
    assert uniform.group_ids is distribution.group_ids
    assert torch.equal(uniform.entry_logits, torch.zeros_like(distribution.entry_logits))
    assert torch.allclose(uniform.probabilities, torch.tensor([2 / 3, 1 / 3]))


def test_sparse_log_categorical_rejects_too_few_dense_classes() -> None:
    distribution = TorchSparseLogCategoricalDistribution(
        group_ids=torch.tensor([0, 2]),
        entry_logits=torch.log(torch.tensor([0.4, 0.6])),
    )

    with pytest.raises(ValueError, match="maximum group id"):
        distribution.to_dense(num_classes=2)


def _torch_modules():
    pytest.importorskip("torch")
    import torch as _torch  # noqa: PLC0415

    return _torch


class TestTorchSparseLogCategoricalAttributes:
    """Quick property checks on the sparse-log categorical distribution."""

    def test_dense_probabilities_shape(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        # A 1-batch distribution with two entries in different groups.
        group_ids = torch.tensor([[0, 1]])
        entry_logits = torch.tensor([[0.0, 0.0]])
        d = TorchSparseLogCategoricalDistribution(group_ids=group_ids, entry_logits=entry_logits)
        # Calling .probabilities should produce a tensor.
        probs = d.probabilities
        assert isinstance(probs, torch.Tensor)


class TestTorchSparseLogCategorical:
    """Torch sparse-log categorical distribution validation."""

    def test_group_ids_must_be_tensor(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(TypeError, match="torch tensor"):
            TorchSparseLogCategoricalDistribution(
                group_ids=[[0, 1]],  # type: ignore[arg-type]
                entry_logits=torch.tensor([[0.0, 0.0]]),
            )

    def test_entry_logits_must_be_tensor(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(TypeError, match="torch tensor"):
            TorchSparseLogCategoricalDistribution(
                group_ids=torch.tensor([[0, 1]]),
                entry_logits=[[0.0, 0.0]],  # type: ignore[arg-type]
            )


class TestValidation:
    """Constructor validation paths."""

    def test_group_ids_int_dtype_required(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(TypeError, match="integer"):
            TorchSparseLogCategoricalDistribution(
                group_ids=torch.tensor([[0.0, 1.0]]),
                entry_logits=torch.tensor([[0.0, 0.0]]),
            )

    def test_entry_logits_float_required(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(TypeError, match="floating point"):
            TorchSparseLogCategoricalDistribution(
                group_ids=torch.tensor([[0, 1]]),
                entry_logits=torch.tensor([[0, 1]]),
            )

    def test_shape_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="identical shapes"):
            TorchSparseLogCategoricalDistribution(
                group_ids=torch.tensor([[0, 1]]),
                entry_logits=torch.tensor([[0.0, 1.0, 2.0]]),
            )

    def test_zero_dim_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="at least one dimension"):
            TorchSparseLogCategoricalDistribution(
                group_ids=torch.tensor(0),
                entry_logits=torch.tensor(0.0),
            )

    def test_empty_sparse_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="at least one sparse"):
            TorchSparseLogCategoricalDistribution(
                group_ids=torch.empty((1, 0), dtype=torch.long),
                entry_logits=torch.empty((1, 0)),
            )

    def test_negative_group_ids_raise(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="non-negative"):
            TorchSparseLogCategoricalDistribution(
                group_ids=torch.tensor([[-1, 1]]),
                entry_logits=torch.tensor([[0.0, 0.0]]),
            )

    def test_nan_logits_raise(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="NaN"):
            TorchSparseLogCategoricalDistribution(
                group_ids=torch.tensor([[0, 1]]),
                entry_logits=torch.tensor([[0.0, float("nan")]]),
            )

    def test_pos_inf_logits_raise(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        with pytest.raises(ValueError, match="positive infinity"):
            TorchSparseLogCategoricalDistribution(
                group_ids=torch.tensor([[0, 1]]),
                entry_logits=torch.tensor([[0.0, float("inf")]]),
            )


class TestProperties:
    def test_num_classes_from_max_group_id(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 4]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        assert d.num_classes == 5

    def test_log_probabilities(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 1]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        # Two distinct groups, equal logits -> log probabilities sum to 0 in log-prob
        # (probabilities sum to 1).
        log_probs = d.log_probabilities
        assert log_probs.shape == (1, 2)
        torch.testing.assert_close(log_probs.exp().sum(dim=-1), torch.tensor([1.0]), atol=1e-5, rtol=1e-5)

    def test_unnormalized_probabilities(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 1]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        unnorm = d.unnormalized_probabilities
        assert unnorm.shape == (1, 2)

    def test_logits_property(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 1]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        logits = d.logits
        assert logits.shape == (1, 2)


class TestUniformLogits:
    def test_returns_zero_logits(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 1]]),
            entry_logits=torch.tensor([[1.0, 2.0]]),
        )
        u = d.uniform_logits()
        torch.testing.assert_close(u.entry_logits, torch.zeros_like(u.entry_logits))


class TestToDense:
    def test_default_num_classes(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 2]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        dense = d.to_dense()
        # max group id 2 -> 3 classes
        assert dense.tensor.shape == (1, 3)

    def test_explicit_num_classes(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 1]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        dense = d.to_dense(num_classes=5)
        assert dense.tensor.shape == (1, 5)

    def test_non_positive_num_classes_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 1]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        with pytest.raises(ValueError, match="must be positive"):
            d.to_dense(num_classes=0)

    def test_too_few_classes_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 5]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        with pytest.raises(ValueError, match="greater than the maximum"):
            d.to_dense(num_classes=2)

    def test_no_finite_logits_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 1]]),
            entry_logits=torch.tensor([[float("-inf"), float("-inf")]]),
        )
        with pytest.raises(ValueError, match="at least one finite"):
            d.to_dense()


class TestSampleAndNumpy:
    def test_sample(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 1]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        s = d.sample(num_samples=4)
        # Returns a TorchSample.
        assert s.tensor.shape == (4, 1)

    def test_numpy(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        d = TorchSparseLogCategoricalDistribution(
            group_ids=torch.tensor([[0, 1]]),
            entry_logits=torch.tensor([[0.0, 0.0]]),
        )
        arr = d.numpy()
        assert arr.shape == (1, 2)


class TestSparseLogCategoricalAttributes:
    """The dense-conversion path on TorchSparseLogCategoricalDistribution."""

    def test_to_dense_via_unnormalized_probabilities(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_sparse_log_categorical import (  # noqa: PLC0415
            TorchSparseLogCategoricalDistribution,
        )

        # Two entries, both belonging to the same group.
        group_ids = torch.tensor([[0, 0]])
        entry_logits = torch.tensor([[0.0, 0.0]])
        d = TorchSparseLogCategoricalDistribution(group_ids=group_ids, entry_logits=entry_logits)
        # Two entries in the same group should yield a non-zero probability.
        probs = d.probabilities
        assert probs is not None
        # The result is a torch tensor.
        assert isinstance(probs, torch.Tensor)
