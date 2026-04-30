"""End-to-end tests for the torch laplace backend."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
laplace_pkg = pytest.importorskip("laplace")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from probly.method.laplace import LaplaceGLMPredictor, LaplaceMCPredictor, laplace  # noqa: E402
from probly.predictor import predict  # noqa: E402
from probly.representation.distribution import CategoricalDistribution, DirichletDistribution  # noqa: E402
from probly.representation.sample import Sample  # noqa: E402
from probly.representer import representer  # noqa: E402
from probly.representer.sampler.torch import LaplaceSampler  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_classifier() -> nn.Module:
    """A tiny 2-layer MLP for 4-way classification on 8-d inputs."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


@pytest.fixture
def tiny_loader() -> DataLoader:
    """A small (32-sample) loader of (x, y) tensors for the tiny classifier."""
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = torch.randint(0, 4, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def tiny_input() -> torch.Tensor:
    """A small input batch matching the tiny classifier's input shape."""
    torch.manual_seed(1)
    return torch.randn(2, 8)


# ---------------------------------------------------------------------------
# End-to-end lifecycle (Task 4: GLM, Task 5: NN)
# ---------------------------------------------------------------------------


def test_glm_construct_fit_predict(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
    tiny_input: torch.Tensor,
) -> None:
    """End-to-end GLM lifecycle: construct, fit, predict."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="glm",
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    assert isinstance(la_pred, LaplaceGLMPredictor)

    la_pred.fit(tiny_loader)
    out = la_pred.predict(tiny_input, link_approx="probit")

    assert out.shape == (tiny_input.shape[0], 4)
    assert torch.allclose(out.sum(dim=-1), torch.ones(tiny_input.shape[0]), atol=1e-5)


def test_glm_sample_returns_posterior_draws(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
    tiny_input: torch.Tensor,
) -> None:
    """``la_glm.sample(x, n_samples=N)`` draws N samples from the GLM Gaussian over logits."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="glm",
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la_pred.fit(tiny_loader)

    samples = la_pred.sample(tiny_input, n_samples=8)
    assert samples.shape[0] == 8
    assert samples.shape[-1] == 4


def test_glm_predict_dirichlet_returns_dirichlet(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
    tiny_input: torch.Tensor,
) -> None:
    """``la_glm.predict_dirichlet(x)`` returns a Dirichlet via the Laplace bridge."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="glm",
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la_pred.fit(tiny_loader)

    out = la_pred.predict_dirichlet(tiny_input)
    assert isinstance(out, DirichletDistribution)
    # alphas must be strictly positive (otherwise the Dirichlet is invalid).
    assert (out.alphas > 0).all()


def test_nn_construct_fit_predict_sample(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
    tiny_input: torch.Tensor,
) -> None:
    """End-to-end NN lifecycle: construct, fit, predict, sample."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="nn",
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    assert isinstance(la_pred, LaplaceMCPredictor)

    la_pred.fit(tiny_loader)

    # pred_type='nn' requires link_approx='mc' (laplace-torch raises ValueError otherwise)
    mean = la_pred.predict(tiny_input, n_samples=8, link_approx="mc")
    assert mean.shape == (tiny_input.shape[0], 4)

    samples = la_pred.sample(tiny_input, n_samples=8)
    assert samples.shape[0] == 8
    assert samples.shape[-1] == 4


# ---------------------------------------------------------------------------
# Unfitted-predict guards (Task 6)
# ---------------------------------------------------------------------------


def test_glm_unfitted_predict_raises(tiny_classifier: nn.Module, tiny_input: torch.Tensor) -> None:
    """Predicting before .fit() raises RuntimeError."""
    la_pred = laplace(tiny_classifier, pred_type="glm", likelihood="classification")

    with pytest.raises(RuntimeError, match=r"Call \.fit\(loader\) before predicting"):
        la_pred.predict(tiny_input)


def test_nn_unfitted_predict_raises(tiny_classifier: nn.Module, tiny_input: torch.Tensor) -> None:
    """Predicting before .fit() raises RuntimeError."""
    la_pred = laplace(tiny_classifier, pred_type="nn", likelihood="classification")

    with pytest.raises(RuntimeError, match=r"Call \.fit\(loader\) before predicting"):
        la_pred.predict(tiny_input)


def test_nn_unfitted_sample_raises(tiny_classifier: nn.Module, tiny_input: torch.Tensor) -> None:
    """Sampling before .fit() raises RuntimeError."""
    la_pred = laplace(tiny_classifier, pred_type="nn", likelihood="classification")

    with pytest.raises(RuntimeError, match=r"Call \.fit\(loader\) before sampling"):
        la_pred.sample(tiny_input)


# ---------------------------------------------------------------------------
# Optimize prior (Task 7)
# ---------------------------------------------------------------------------


def test_optimize_prior_changes_precision(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
) -> None:
    """``.fit(..., optimize_prior=True)`` updates ``la.prior_precision``."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="glm",
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    # prior_precision is a float before fitting; capture as scalar
    initial_prior = float(la_pred.la.prior_precision)

    la_pred.fit(tiny_loader, optimize_prior=True, n_steps=10)

    # After optimize_prior, prior_precision may be a float or tensor
    final_prior = float(la_pred.la.prior_precision)
    assert initial_prior != final_prior, "optimize_prior=True should change la.prior_precision"


# ---------------------------------------------------------------------------
# Configuration pass-through (Task 8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subset_of_weights", "hessian_structure"),
    [
        ("all", "diag"),
        ("last_layer", "kron"),
    ],
)
def test_config_pass_through(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
    tiny_input: torch.Tensor,
    subset_of_weights: str,
    hessian_structure: str,
) -> None:
    """Different ``subset_of_weights x hessian_structure`` combinations work end-to-end."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="glm",
        likelihood="classification",
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
    )
    la_pred.fit(tiny_loader)
    out = la_pred.predict(tiny_input, link_approx="probit")
    assert out.shape == (tiny_input.shape[0], 4)


# ---------------------------------------------------------------------------
# probly distribution integration (top-level predict() flexdispatch)
# ---------------------------------------------------------------------------


def test_glm_predict_returns_categorical_distribution(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
    tiny_input: torch.Tensor,
) -> None:
    """``predict(la_pred, x)`` wraps GLM classification output in a CategoricalDistribution."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="glm",
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la_pred.fit(tiny_loader)

    out = predict(la_pred, tiny_input, link_approx="probit")
    assert isinstance(out, CategoricalDistribution)


def test_mc_predict_returns_categorical_distribution(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
    tiny_input: torch.Tensor,
) -> None:
    """``predict(la_pred, x)`` wraps MC classification output in a CategoricalDistribution."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="nn",
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la_pred.fit(tiny_loader)

    out = predict(la_pred, tiny_input, n_samples=8, link_approx="mc")
    assert isinstance(out, CategoricalDistribution)


# ---------------------------------------------------------------------------
# Representer integration (representer() flexdispatch)
# ---------------------------------------------------------------------------


def test_glm_representer_returns_laplace_sampler(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
    tiny_input: torch.Tensor,
) -> None:
    """``representer(la_glm, num_samples=N)`` returns a LaplaceSampler producing a Sample."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="glm",
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la_pred.fit(tiny_loader)

    rep = representer(la_pred, num_samples=8)
    assert isinstance(rep, LaplaceSampler)
    assert rep.num_samples == 8

    out = rep.represent(tiny_input)
    assert isinstance(out, Sample)
    assert out.size(out.sample_axis) == 8


def test_mc_representer_returns_laplace_sampler(
    tiny_classifier: nn.Module,
    tiny_loader: DataLoader,
    tiny_input: torch.Tensor,
) -> None:
    """``representer(la_mc, num_samples=N)`` returns a LaplaceSampler producing a Sample."""
    la_pred = laplace(
        tiny_classifier,
        pred_type="nn",
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la_pred.fit(tiny_loader)

    rep = representer(la_pred, num_samples=8)
    assert isinstance(rep, LaplaceSampler)
    assert rep.num_samples == 8

    out = rep.represent(tiny_input)
    assert isinstance(out, Sample)
    # The 8 posterior draws live along ``out.sample_axis`` of the underlying
    # categorical-probabilities tensor.
    assert out.size(out.sample_axis) == 8
