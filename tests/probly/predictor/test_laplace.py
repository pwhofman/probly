"""End-to-end tests for the laplace-torch predict() dispatch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
laplace_pkg = pytest.importorskip("laplace")

from laplace import Laplace  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from probly.predictor import predict  # noqa: E402
from probly.representation.distribution import CategoricalDistribution  # noqa: E402


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


def test_predict_returns_categorical_distribution_glm(fitted_glm_la, tiny_input: torch.Tensor) -> None:
    """``predict(la, x)`` defaults to pred_type='glm' and returns a CategoricalDistribution."""
    out = predict(fitted_glm_la, tiny_input, link_approx="probit")
    assert isinstance(out, CategoricalDistribution)


def test_predict_returns_categorical_distribution_nn(fitted_glm_la, tiny_input: torch.Tensor) -> None:
    """``predict(la, x, pred_type='nn', ...)`` returns a CategoricalDistribution."""
    out = predict(fitted_glm_la, tiny_input, pred_type="nn", link_approx="mc", n_samples=8)
    assert isinstance(out, CategoricalDistribution)


def test_predict_default_pred_type_is_glm(fitted_glm_la, tiny_input: torch.Tensor) -> None:
    """Calling predict without an explicit pred_type uses glm (laplace-torch's default)."""
    out_default = predict(fitted_glm_la, tiny_input, link_approx="probit")
    out_explicit = predict(fitted_glm_la, tiny_input, pred_type="glm", link_approx="probit")
    # Both invocations should produce the same closed-form result.
    assert torch.allclose(out_default.unnormalized_probabilities, out_explicit.unnormalized_probabilities)


def test_predict_regression_raises_not_implemented(tiny_classifier: nn.Module) -> None:
    """``likelihood='regression'`` is not yet wrapped; predict() raises with a clear message."""
    # Convert labels to floats so laplace-torch accepts regression.
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    reg_loader = DataLoader(TensorDataset(x, y), batch_size=8)
    la = Laplace(tiny_classifier, "regression", subset_of_weights="last_layer", hessian_structure="kron")
    la.fit(reg_loader)

    with pytest.raises(NotImplementedError, match="likelihood='classification'"):
        predict(la, torch.randn(2, 8))
