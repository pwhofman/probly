"""Tests for the torch credal relative likelihood training function."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from functools import partial  # noqa: E402

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from probly.method.credal_relative_likelihood import credal_relative_likelihood  # noqa: E402
from probly.method.credal_relative_likelihood.torch import train_credal_relative_likelihood  # noqa: E402


def _blobs_loader() -> DataLoader:
    """Three well-separated 2d classes, deterministic order."""
    torch.manual_seed(0)
    centers = torch.tensor([[-4.0, 0.0], [4.0, 0.0], [0.0, 5.0]])
    inputs = torch.cat([center + torch.randn(30, 2) for center in centers])
    targets = torch.arange(3).repeat_interleave(30)
    return DataLoader(TensorDataset(inputs, targets), batch_size=32, shuffle=False)


def _predictor(num_members: int = 4):
    torch.manual_seed(1)
    base = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
    return credal_relative_likelihood(base, num_members=num_members, tobias_value=3, predictor_type="logit_classifier")


class TestTrainCredalRelativeLikelihood:
    """Reference member plus relative-likelihood-constrained members."""

    def test_alpha_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            train_credal_relative_likelihood(_predictor(), _blobs_loader(), alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            train_credal_relative_likelihood(_predictor(), _blobs_loader(), alpha=1.5)

    def test_reference_trains_full_epochs_and_members_carry_thresholds(self) -> None:
        records: list[dict[str, float]] = []
        train_credal_relative_likelihood(_predictor(), _blobs_loader(), alpha=0.5, epochs=2, on_epoch=records.append)
        reference_records = [r for r in records if r["member"] == 0.0]
        member_records = [r for r in records if r["member"] > 0.0]
        assert len(reference_records) == 2
        assert all("threshold" not in r and "relative_likelihood" not in r for r in reference_records)
        assert all("threshold" in r and "relative_likelihood" in r for r in member_records)
        thresholds = sorted({r["threshold"] for r in member_records})
        assert thresholds == pytest.approx([0.5, 0.5 + 0.5 / 3, 0.5 + 1.0 / 3])

    def test_members_stop_once_target_reached(self) -> None:
        records: list[dict[str, float]] = []
        train_credal_relative_likelihood(
            _predictor(),
            _blobs_loader(),
            alpha=0.5,
            epochs=25,
            optimizer_factory=partial(torch.optim.Adam, lr=0.02),
            on_epoch=records.append,
        )
        for member_index in (1.0, 2.0, 3.0):
            member_records = [r for r in records if r["member"] == member_index]
            assert len(member_records) < 25
            assert member_records[-1]["relative_likelihood"] >= member_records[-1]["threshold"]

    def test_user_hook_can_stop_members_before_their_target(self) -> None:
        records: list[dict[str, float]] = []

        def stop_non_reference(metrics: dict[str, float]) -> bool:
            records.append(metrics)
            return metrics["member"] >= 1.0

        train_credal_relative_likelihood(_predictor(), _blobs_loader(), epochs=3, on_epoch=stop_non_reference)
        counts = {member: len([r for r in records if r["member"] == member]) for member in (0.0, 1.0, 2.0, 3.0)}
        assert counts == {0.0: 3, 1.0: 1, 2.0: 1, 3.0: 1}

    def test_single_member_trains_reference_only(self) -> None:
        records: list[dict[str, float]] = []
        train_credal_relative_likelihood(_predictor(num_members=1), _blobs_loader(), epochs=2, on_epoch=records.append)
        assert {r["member"] for r in records} == {0.0}
