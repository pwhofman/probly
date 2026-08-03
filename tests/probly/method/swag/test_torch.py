"""Torch-specific tests for the SWAG method."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.nn.utils import parameters_to_vector, vector_to_parameters  # noqa: E402

from probly.method.swag import SWAGPredictor, collect_swag, swag  # noqa: E402
from probly.method.swag.torch import TorchSWAGPredictor  # noqa: E402
from probly.quantification import quantify  # noqa: E402
from probly.representer import representer  # noqa: E402
from probly.representer.sampler._common import Sampler  # noqa: E402


@pytest.fixture
def linear_model() -> nn.Module:
    """Small MLP classifier."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )


def make_swag(
    model: nn.Module, max_rank: int = 20, scale: float = 0.5, predictor_type: str | None = None
) -> TorchSWAGPredictor:
    """Apply swag and narrow the return type to the torch wrapper."""
    swag_model = swag(model, max_rank=max_rank, scale=scale, predictor_type=predictor_type)
    assert isinstance(swag_model, TorchSWAGPredictor)
    return swag_model


def collect_with_weights(model: TorchSWAGPredictor, weights: torch.Tensor) -> None:
    """Load a flat weight vector into the wrapped model and collect it."""
    with torch.no_grad():
        vector_to_parameters(weights, model.model.parameters())
    collect_swag(model)


class TestTransformation:
    """Tests for the swag transformation itself."""

    def test_returns_swag_predictor(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model)
        assert isinstance(model, TorchSWAGPredictor)
        assert isinstance(model, SWAGPredictor)

    def test_original_model_is_not_mutated(self, linear_model: nn.Module) -> None:
        original = parameters_to_vector(linear_model.parameters()).detach().clone()
        model = make_swag(linear_model)
        assert model.model is not linear_model

        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        loss = model(torch.randn(6, 4)).square().sum()
        loss.backward()
        opt.step()

        assert torch.equal(parameters_to_vector(linear_model.parameters()), original)

    def test_forward_shape(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model)
        out = model(torch.randn(6, 4))
        assert out.shape == (6, 3)

    def test_buffers_are_in_state_dict(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        state = model.state_dict()
        for name in ("mean", "sq_mean", "deviations", "num_collected"):
            assert name in state


class TestCollect:
    """Tests for the collection of SWAG statistics."""

    def test_moments_match_manual_computation(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        snapshots = [torch.randn_like(model.mean) for _ in range(3)]
        for weights in snapshots:
            collect_with_weights(model, weights)

        stacked = torch.stack(snapshots)
        assert int(model.num_collected) == 3
        torch.testing.assert_close(model.mean, stacked.mean(dim=0))
        torch.testing.assert_close(model.sq_mean, stacked.square().mean(dim=0))

    def test_deviations_use_running_mean(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        snapshots = [torch.randn_like(model.mean) for _ in range(3)]
        for weights in snapshots:
            collect_with_weights(model, weights)

        stacked = torch.stack(snapshots)
        for i in range(3):
            running_mean = stacked[: i + 1].mean(dim=0)
            torch.testing.assert_close(model.deviations[i], snapshots[i] - running_mean)

    def test_deviations_keep_last_max_rank_snapshots(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model, max_rank=2)
        snapshots = [torch.randn_like(model.mean) for _ in range(3)]
        for weights in snapshots:
            collect_with_weights(model, weights)

        stacked = torch.stack(snapshots)
        assert model.deviations.shape[0] == 2
        torch.testing.assert_close(model.deviations[0], snapshots[1] - stacked[:2].mean(dim=0))
        torch.testing.assert_close(model.deviations[1], snapshots[2] - stacked.mean(dim=0))


class TestSampling:
    """Tests for sampling weights from the SWAG posterior."""

    def test_sample_before_collect_raises(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model)
        with pytest.raises(RuntimeError, match="No weight snapshots have been collected yet"):
            model.sample_parameters()

    def test_sample_with_zero_scale_loads_mean(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        for _ in range(3):
            collect_with_weights(model, torch.randn_like(model.mean))

        model.sample_parameters(scale=0.0)
        torch.testing.assert_close(parameters_to_vector(model.model.parameters()), model.mean)

    def test_load_mean_parameters(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        for _ in range(3):
            collect_with_weights(model, torch.randn_like(model.mean))

        model.load_mean_parameters()
        torch.testing.assert_close(parameters_to_vector(model.model.parameters()), model.mean)

    def test_sample_perturbs_weights(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model, max_rank=5)
        for _ in range(3):
            collect_with_weights(model, torch.randn_like(model.mean))

        model.sample_parameters()
        first = parameters_to_vector(model.model.parameters()).clone()
        model.sample_parameters()
        second = parameters_to_vector(model.model.parameters())
        assert not torch.equal(first, second)

    def test_diagonal_only_sampling(self, linear_model: nn.Module) -> None:
        model = make_swag(linear_model, max_rank=0)
        for _ in range(3):
            collect_with_weights(model, torch.randn_like(model.mean))

        model.sample_parameters()
        assert not torch.equal(parameters_to_vector(model.model.parameters()), model.mean)


class TestRepresenter:
    """End-to-end tests through the representer and quantification."""

    @pytest.fixture
    def collected_model(self, linear_model: nn.Module) -> TorchSWAGPredictor:
        model = make_swag(linear_model, max_rank=5, predictor_type="logit_classifier")
        for _ in range(4):
            collect_with_weights(model, torch.randn_like(model.mean))
        return model

    def test_representer_dispatches_to_sampler(self, collected_model: TorchSWAGPredictor) -> None:
        rep = representer(collected_model, num_samples=7)
        assert isinstance(rep, Sampler)

    def test_represent_and_quantify(self, collected_model: TorchSWAGPredictor) -> None:
        rep = representer(collected_model, num_samples=7)
        with torch.no_grad():
            out = rep.represent(torch.randn(5, 4))
        decomp = quantify(out)
        assert decomp["total"].shape == (5,)

    def test_sampling_restores_weights_and_flag(self, collected_model: TorchSWAGPredictor) -> None:
        before = parameters_to_vector(collected_model.model.parameters()).clone()
        rep = representer(collected_model, num_samples=3)
        with torch.no_grad():
            rep.represent(torch.randn(5, 4))
        after = parameters_to_vector(collected_model.model.parameters())
        assert torch.equal(before, after)
        assert collected_model.sampling is False
