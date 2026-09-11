"""Tests for the generic torch training blocks in probly.train.torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from collections.abc import Iterator  # noqa: E402
from functools import partial  # noqa: E402

from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from probly.train.torch import train_model  # noqa: E402


def _separable_loader(batch_size: int = 16) -> DataLoader:
    """Two well-separated 2d classes, deterministic order."""
    torch.manual_seed(0)
    inputs = torch.cat([torch.randn(24, 2) - 3.0, torch.randn(24, 2) + 3.0])
    targets = torch.arange(2).repeat_interleave(24)
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)


def _ragged_loader() -> DataLoader:
    """Four samples in batches of sizes 3 and 1, so naive and sample-weighted means differ."""
    inputs = torch.zeros(4, 2)
    targets = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(inputs, targets), batch_size=3, shuffle=False)


def _batch_size_loss(output: Tensor, targets: Tensor) -> Tensor:
    """Differentiable loss whose value equals the batch size."""
    return output.sum() * 0.0 + float(targets.shape[0])


def _make_model() -> nn.Module:
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))


class TestTrainModel:
    """The shared supervised loop: hook contract, metric semantics, and knobs."""

    def test_training_reduces_loss_and_returns_eval_model(self) -> None:
        records: list[dict[str, float]] = []
        model = _make_model()
        loss_fn = nn.functional.cross_entropy
        trained = train_model(
            model,
            _separable_loader(),
            loss_fn,
            epochs=10,
            optimizer_factory=partial(torch.optim.SGD, lr=0.5),
            on_epoch=records.append,
        )
        assert trained is model
        assert not model.training
        assert records[-1]["running_loss"] < records[0]["running_loss"]

    def test_hook_called_once_per_epoch_with_expected_keys(self) -> None:
        records: list[dict[str, float]] = []
        train_model(_make_model(), _separable_loader(), nn.functional.cross_entropy, epochs=3, on_epoch=records.append)
        assert [r["epoch"] for r in records] == [0.0, 1.0, 2.0]
        assert all(set(r) == {"epoch", "running_loss"} for r in records)

    def test_hook_returning_true_stops_training(self) -> None:
        records: list[dict[str, float]] = []
        train_model(
            _make_model(),
            _separable_loader(),
            nn.functional.cross_entropy,
            epochs=10,
            on_epoch=lambda m: records.append(m) or True,
        )
        assert len(records) == 1

    def test_extra_metrics_merged_and_not_mutated(self) -> None:
        records: list[dict[str, float]] = []
        extra = {"member": 3.0}
        train_model(
            _make_model(),
            _separable_loader(),
            nn.functional.cross_entropy,
            epochs=2,
            on_epoch=records.append,
            extra_metrics=extra,
        )
        assert all(r["member"] == 3.0 for r in records)
        assert extra == {"member": 3.0}

    def test_running_loss_is_sample_weighted(self) -> None:
        # Batch sizes 3 and 1: the sample-weighted mean of the batch-size loss is
        # (3*3 + 1*1) / 4 = 2.5, whereas a mean of batch means would give 2.0.
        records: list[dict[str, float]] = []
        train_model(nn.Linear(2, 2), _ragged_loader(), _batch_size_loss, epochs=1, on_epoch=records.append)
        assert records[0]["running_loss"] == pytest.approx(2.5)

    def test_val_loss_is_sample_weighted_and_uses_same_loss(self) -> None:
        records: list[dict[str, float]] = []
        train_model(
            nn.Linear(2, 2),
            _ragged_loader(),
            _batch_size_loss,
            val_loader=_ragged_loader(),
            epochs=1,
            on_epoch=records.append,
        )
        assert records[0]["val_loss"] == pytest.approx(2.5)

    def test_validation_does_not_affect_training(self) -> None:
        loss_fn = nn.functional.cross_entropy
        plain = train_model(_make_model(), _separable_loader(), loss_fn, epochs=3)
        validated = train_model(_make_model(), _separable_loader(), loss_fn, val_loader=_separable_loader(), epochs=3)
        for p1, p2 in zip(plain.state_dict().values(), validated.state_dict().values(), strict=True):
            assert torch.equal(p1, p2)

    def test_scheduler_steps_once_per_epoch(self) -> None:
        optimizers: list[torch.optim.Optimizer] = []

        def factory(params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
            optimizer = torch.optim.SGD(params, lr=0.1)
            optimizers.append(optimizer)
            return optimizer

        train_model(
            _make_model(),
            _separable_loader(),
            nn.functional.cross_entropy,
            epochs=3,
            optimizer_factory=factory,
            scheduler_factory=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.5),
        )
        assert len(optimizers) == 1
        assert optimizers[0].param_groups[0]["lr"] == pytest.approx(0.1 * 0.5**3)

    def test_exactly_one_fresh_gradient_step_per_batch(self) -> None:
        # A weight-1 linear model with loss sum(w * x) on all-ones inputs has gradient 2 per
        # size-2 batch, so 3 epochs over 2 batches with plain SGD move the weight by exactly
        # 6 * lr * 2. Leaked gradient accumulation or extra steps break the closed form.
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.zeros(4)), batch_size=2, shuffle=False)
        train_model(
            model,
            loader,
            lambda output, _targets: output.sum(),
            epochs=3,
            optimizer_factory=partial(torch.optim.SGD, lr=0.1),
        )
        assert model.weight.item() == pytest.approx(1.0 - 3 * 2 * 0.1 * 2.0, abs=1e-6)

    def test_train_mode_during_training_and_eval_mode_during_validation(self) -> None:
        model = _make_model()
        observed: list[tuple[bool, bool]] = []

        def spying_loss(output: Tensor, targets: Tensor) -> Tensor:
            observed.append((model.training, torch.is_grad_enabled()))
            return nn.functional.cross_entropy(output, targets)

        loader = _separable_loader(batch_size=24)  # two training batches per epoch
        val_loader = _separable_loader(batch_size=48)  # one validation batch per epoch
        train_model(model, loader, spying_loss, val_loader=val_loader, epochs=2)
        assert observed == [(True, True), (True, True), (False, False)] * 2

    def test_generic_loss_supports_regression(self) -> None:
        torch.manual_seed(2)
        inputs = torch.randn(32, 2)
        targets = inputs.sum(dim=1, keepdim=True)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=8, shuffle=False)
        records: list[dict[str, float]] = []
        train_model(
            nn.Linear(2, 1),
            loader,
            nn.functional.mse_loss,
            epochs=20,
            optimizer_factory=partial(torch.optim.SGD, lr=0.1),
            on_epoch=records.append,
        )
        assert records[-1]["running_loss"] < 0.1 * records[0]["running_loss"]

    def test_hook_exceptions_propagate(self) -> None:
        def failing_hook(_metrics: dict[str, float]) -> bool:
            msg = "user hook failure"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="user hook failure"):
            train_model(_make_model(), _separable_loader(), nn.functional.cross_entropy, on_epoch=failing_hook)

    def test_empty_train_loader_raises(self) -> None:
        empty = DataLoader(TensorDataset(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)), batch_size=4)
        with pytest.raises(ValueError, match="train_loader"):
            train_model(_make_model(), empty, nn.functional.cross_entropy)

    def test_empty_val_loader_raises(self) -> None:
        empty = DataLoader(TensorDataset(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)), batch_size=4)
        with pytest.raises(ValueError, match="validation loader"):
            train_model(_make_model(), _separable_loader(), nn.functional.cross_entropy, val_loader=empty)

    def test_epochs_zero_is_a_noop(self) -> None:
        model = _make_model()
        before = [p.clone() for p in model.parameters()]
        records: list[dict[str, float]] = []
        train_model(model, _separable_loader(), nn.functional.cross_entropy, epochs=0, on_epoch=records.append)
        assert records == []
        assert not model.training
        assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))
