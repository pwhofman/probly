"""Tests for the torch credal DRO training function."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from collections.abc import Iterator  # noqa: E402

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from probly.method.credal_dro import credal_dro  # noqa: E402
from probly.method.credal_dro.torch import train_credal_dro  # noqa: E402
from probly.representation.credal_set import ProbabilityIntervalsCredalSet  # noqa: E402
from probly.representer import representer  # noqa: E402


def _blobs_loader() -> DataLoader:
    """Three well-separated 2d classes, deterministic order."""
    torch.manual_seed(0)
    centers = torch.tensor([[-4.0, 0.0], [4.0, 0.0], [0.0, 5.0]])
    inputs = torch.cat([center + torch.randn(30, 2) for center in centers])
    targets = torch.arange(3).repeat_interleave(30)
    return DataLoader(TensorDataset(inputs, targets), batch_size=32, shuffle=False)


def _predictor(num_members: int = 3):
    torch.manual_seed(1)
    base = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
    return credal_dro(base, num_members=num_members, predictor_type="logit_classifier")


class TestTrainCredalDro:
    """Per-member CVaR training at the Eq. 8 delta schedule."""

    def test_members_train_at_schedule_deltas(self) -> None:
        records: list[dict[str, float]] = []
        train_credal_dro(_predictor(), _blobs_loader(), delta_g=0.5, epochs=2, on_epoch=records.append)
        assert {(r["member"], r["delta"]) for r in records} == {(0.0, 0.5), (1.0, 0.75), (2.0, 1.0)}
        assert len(records) == 3 * 2  # one record per member and epoch

    def test_every_member_learns(self) -> None:
        predictor = _predictor()
        initial = [[p.clone() for p in member.parameters()] for member in predictor]
        records: list[dict[str, float]] = []
        train_credal_dro(predictor, _blobs_loader(), epochs=4, on_epoch=records.append)
        for member, before in zip(predictor, initial, strict=True):
            assert any(not torch.equal(p, q) for p, q in zip(member.parameters(), before, strict=True))
        for member_index in range(3):
            member_losses = [r["running_loss"] for r in records if r["member"] == member_index]
            assert member_losses[-1] < member_losses[0]

    def test_trained_predictor_yields_probability_intervals(self) -> None:
        predictor = train_credal_dro(_predictor(), _blobs_loader(), epochs=1)
        output = representer(predictor).predict(torch.randn(5, 2))
        assert isinstance(output, ProbabilityIntervalsCredalSet)

    def test_hook_stop_applies_per_member(self) -> None:
        records: list[dict[str, float]] = []
        train_credal_dro(_predictor(), _blobs_loader(), epochs=10, on_epoch=lambda m: records.append(m) or True)
        assert [r["member"] for r in records] == [0.0, 1.0, 2.0]

    def test_optimizer_factory_built_once_per_member(self) -> None:
        optimizers: list[torch.optim.Optimizer] = []

        def factory(params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
            optimizer = torch.optim.SGD(params, lr=0.1)
            optimizers.append(optimizer)
            return optimizer

        train_credal_dro(_predictor(), _blobs_loader(), epochs=1, optimizer_factory=factory)
        assert len(optimizers) == 3

    def test_val_loss_reported_for_every_member(self) -> None:
        records: list[dict[str, float]] = []
        train_credal_dro(_predictor(), _blobs_loader(), val_loader=_blobs_loader(), epochs=2, on_epoch=records.append)
        assert all("val_loss" in r for r in records)
