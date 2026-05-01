"""End-to-end tests for the LaplaceSampler representer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
laplace_pkg = pytest.importorskip("laplace")

from laplace import Laplace  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from probly.method.laplace.laplace import LaplaceSampler  # noqa: E402
from probly.representation.distribution import CategoricalDistribution  # noqa: E402
from probly.representation.sample import Sample  # noqa: E402
from probly.representer import representer  # noqa: E402


@pytest.fixture
def tiny_classifier() -> nn.Module:
    """A tiny LeNet-style classifier ending in nn.Linear (required by last_layer Laplace)."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


@pytest.fixture
def tiny_loader() -> DataLoader:
    """Small loader for fitting the Laplace posterior."""
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = torch.randint(0, 4, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def tiny_input() -> torch.Tensor:
    """Test-time input batch."""
    torch.manual_seed(1)
    return torch.randn(2, 8)


@pytest.fixture
def fitted_glm_la(tiny_classifier: nn.Module, tiny_loader: DataLoader):
    """A fitted Laplace approximation in classification + last_layer + KFAC mode."""
    la = Laplace(
        tiny_classifier,
        "classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la.fit(tiny_loader)
    return la


def test_representer_returns_laplace_sampler_glm(fitted_glm_la, tiny_input: torch.Tensor) -> None:
    """``representer(la, num_samples=N)`` returns a LaplaceSampler in GLM mode (default)."""
    rep = representer(fitted_glm_la, num_samples=8)
    assert isinstance(rep, LaplaceSampler)
    assert rep.num_samples == 8
    assert rep.pred_type == "glm"

    out = rep.represent(tiny_input)
    assert isinstance(out, Sample)
    assert out.size(out.sample_axis) == 8


def test_representer_returns_laplace_sampler_nn(fitted_glm_la, tiny_input: torch.Tensor) -> None:
    """``representer(la, num_samples=N, pred_type='nn')`` returns a LaplaceSampler in MC mode."""
    rep = representer(fitted_glm_la, num_samples=8, pred_type="nn")
    assert isinstance(rep, LaplaceSampler)
    assert rep.pred_type == "nn"

    out = rep.represent(tiny_input)
    assert isinstance(out, Sample)
    assert out.size(out.sample_axis) == 8


def test_laplace_sampler_pred_type_default_is_glm(fitted_glm_la) -> None:
    """When no pred_type is given, the sampler defaults to glm."""
    rep = representer(fitted_glm_la, num_samples=4)
    assert rep.pred_type == "glm"


def test_laplace_sampler_regression_raises(tiny_classifier: nn.Module) -> None:
    """LaplaceSampler.represent on a regression Laplace raises NotImplementedError."""
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    reg_loader = DataLoader(TensorDataset(x, y), batch_size=8)
    la = Laplace(tiny_classifier, "regression", subset_of_weights="last_layer", hessian_structure="kron")
    la.fit(reg_loader)

    rep = representer(la, num_samples=4)
    with pytest.raises(NotImplementedError, match="likelihood='classification'"):
        rep.represent(torch.randn(2, 8))


def test_laplace_sampler_yields_categorical_distributions_per_draw(fitted_glm_la, tiny_input: torch.Tensor) -> None:
    """Each posterior draw materializes as a CategoricalDistribution before aggregation."""
    rep = representer(fitted_glm_la, num_samples=4)
    drawn = list(rep._predict(tiny_input))  # noqa: SLF001
    assert len(drawn) == 4
    assert all(isinstance(d, CategoricalDistribution) for d in drawn)
