"""Tests for the optional GPyTorch bindings."""

from __future__ import annotations

import subprocess
import sys
from typing import Any, cast

import pytest

pytest.importorskip("torch")
pytest.importorskip("gpytorch")

import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood, MultitaskGaussianLikelihood, SoftmaxLikelihood
import torch

from probly.calibrator import calibrate
from probly.method.conformal import conformal_absolute_error
from probly.metrics import coverage
from probly.predictor import GaussianDistributionPredictor, predict
from probly.quantification import SecondOrderEntropyDecomposition, SecondOrderVarianceDecomposition
from probly.quantification.measure.variance import variance
from probly.representation.distribution.torch_bernoulli import TorchBernoulliDistributionSample
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.distribution.torch_gaussian import (
    TorchGaussianDistribution,
    TorchGaussianDistributionSample,
)
from probly.representer import representer

NUM_TRAIN = 24
NUM_CLASSES = 3


class _ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: GaussianLikelihood) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _MultitaskExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: MultitaskGaussianLikelihood) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=2)
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1)

    def forward(self, x: torch.Tensor) -> MultitaskMultivariateNormal:
        return MultitaskMultivariateNormal(self.mean_module(x), self.covar_module(x))


class _MultitaskSVGP(gpytorch.models.ApproximateGP):
    """One latent function per class, the GPyTorch multiclass recipe."""

    def __init__(self, inducing_points: torch.Tensor, num_latents: int) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution),
            num_tasks=num_latents,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]))

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _ScalarSVGP(gpytorch.models.ApproximateGP):
    """Single latent function, used with a Bernoulli likelihood."""

    def __init__(self, inducing_points: torch.Tensor) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2))
        super().__init__(gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution))
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


@pytest.fixture
def train_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, NUM_TRAIN)
    y = torch.sin(2.0 * torch.pi * x) + 0.1 * torch.randn(NUM_TRAIN)
    return x, y


@pytest.fixture
def exact_gp(train_data: tuple[torch.Tensor, torch.Tensor]) -> _ExactGP:
    x, y = train_data
    model = _ExactGP(x, y, GaussianLikelihood())
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        loss = -mll(model(x), y)  # ty: ignore[unsupported-operator]
        loss.backward()
        optimizer.step()
    return model.eval()


def test_exact_gp_is_gaussian_distribution_predictor(exact_gp: _ExactGP) -> None:
    assert isinstance(exact_gp, GaussianDistributionPredictor)


def test_exact_gp_predict_returns_predictive_gaussian(exact_gp: _ExactGP) -> None:
    x = torch.tensor([0.25, 0.5, 1.5])
    with torch.no_grad():
        prediction = predict(exact_gp, x)
        latent = exact_gp(x)
        predictive = cast("Any", exact_gp).likelihood(latent)

    assert isinstance(prediction, TorchGaussianDistribution)
    assert torch.allclose(prediction.mean, predictive.mean)
    assert torch.allclose(prediction.var, predictive.variance)
    assert torch.all(prediction.var > latent.variance)


def test_approximate_gp_predict_returns_latent_posterior() -> None:
    torch.manual_seed(0)
    model = _MultitaskSVGP(torch.rand(6, 2), NUM_CLASSES).eval()
    x = torch.rand(5, 2)
    with torch.no_grad():
        prediction = predict(model, x)
        latent = model(x)

    assert isinstance(prediction, TorchGaussianDistribution)
    assert prediction.mean.shape == (5, NUM_CLASSES)
    assert torch.allclose(prediction.var, latent.variance)


def test_multitask_exact_gp_predict_keeps_task_axis() -> None:
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, NUM_TRAIN)
    y = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
    model = _MultitaskExactGP(x, y, MultitaskGaussianLikelihood(num_tasks=2)).eval()
    with torch.no_grad():
        prediction = predict(model, torch.tensor([0.1, 0.9]))

    assert prediction.mean.shape == (2, 2)
    assert prediction.var.shape == (2, 2)


def test_predict_keeps_input_gradients(exact_gp: _ExactGP) -> None:
    x = torch.tensor([0.3, 1.2], requires_grad=True)

    total_variance = variance(predict(exact_gp, x)).sum()
    (gradient,) = torch.autograd.grad(total_variance, x)

    assert gradient.shape == x.shape
    assert torch.all(torch.isfinite(gradient))


def test_stacked_exact_gps_form_gaussian_sample(train_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = train_data
    members = []
    for lengthscale in (0.05, 0.3):
        model = _ExactGP(x, y, GaussianLikelihood()).eval()
        model.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        members.append(model)
    test_x = torch.tensor([0.5, 1.5])
    with torch.no_grad():
        predictions = [predict(member, test_x) for member in members]
    sample = TorchGaussianDistributionSample(
        TorchGaussianDistribution(
            mean=torch.stack([p.mean for p in predictions]),
            var=torch.stack([p.var for p in predictions]),
        ),
        sample_dim=0,
    )

    decomposition = SecondOrderVarianceDecomposition(sample)

    assert torch.allclose(decomposition.total, decomposition.aleatoric + decomposition.epistemic)
    assert torch.all(decomposition.epistemic >= 0.0)


def test_gpytorch_is_not_imported_eagerly() -> None:
    code = """
import sys
import probly
import probly.integrations
import probly.predictor
import probly.representer
assert 'gpytorch' not in sys.modules, 'gpytorch was imported eagerly'
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr


@pytest.fixture
def softmax_svgp() -> tuple[_MultitaskSVGP, SoftmaxLikelihood]:
    torch.manual_seed(0)
    x = torch.randn(NUM_TRAIN, 2)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()  # class labels 0, 1, 2
    model = _MultitaskSVGP(x[:6], NUM_CLASSES)
    likelihood = SoftmaxLikelihood(num_features=NUM_CLASSES, num_classes=NUM_CLASSES)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=NUM_TRAIN)
    optimizer = torch.optim.Adam([*model.parameters(), *likelihood.parameters()], lr=0.05)
    model.train()
    likelihood.train()
    for _ in range(10):
        optimizer.zero_grad()
        loss = -mll(model(x), y)  # ty: ignore[unsupported-operator]
        loss.backward()
        optimizer.step()
    return model.eval(), likelihood.eval()


def test_softmax_representer_returns_categorical_sample(softmax_svgp: tuple[_MultitaskSVGP, SoftmaxLikelihood]) -> None:
    model, likelihood = softmax_svgp
    rep = representer(model, num_samples=8, likelihood=likelihood, sample_axis=0)
    with torch.no_grad():
        sample = rep.represent(torch.randn(4, 2))

    assert isinstance(sample, TorchCategoricalDistributionSample)
    assert sample.sample_dim == 0
    probabilities = sample.tensor.probabilities
    assert probabilities.shape == (8, 4, NUM_CLASSES)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(8, 4))


def test_softmax_representer_default_sample_axis_is_last_batch_axis(
    softmax_svgp: tuple[_MultitaskSVGP, SoftmaxLikelihood],
) -> None:
    model, likelihood = softmax_svgp
    with torch.no_grad():
        sample = representer(model, num_samples=8, likelihood=likelihood).represent(torch.randn(4, 2))

    assert sample.sample_dim == 1
    assert sample.tensor.probabilities.shape == (4, 8, NUM_CLASSES)


def test_softmax_representer_entropy_decomposition(softmax_svgp: tuple[_MultitaskSVGP, SoftmaxLikelihood]) -> None:
    model, likelihood = softmax_svgp
    with torch.no_grad():
        sample = representer(model, num_samples=16, likelihood=likelihood).represent(torch.randn(5, 2))

    decomposition = SecondOrderEntropyDecomposition(sample)

    assert decomposition.total.shape == (5,)
    assert torch.all(torch.isfinite(decomposition.total))
    assert torch.all(decomposition.epistemic >= -1e-6)


def test_softmax_representer_keeps_input_gradients(softmax_svgp: tuple[_MultitaskSVGP, SoftmaxLikelihood]) -> None:
    model, likelihood = softmax_svgp
    x = torch.randn(3, 2, requires_grad=True)

    sample = representer(model, num_samples=8, likelihood=likelihood).represent(x)
    (gradient,) = torch.autograd.grad(sample.tensor.probabilities.sum(), x)

    assert torch.all(torch.isfinite(gradient))


def test_bernoulli_representer_returns_bernoulli_sample() -> None:
    torch.manual_seed(0)
    model = _ScalarSVGP(torch.randn(6, 2)).eval()
    rep = representer(model, num_samples=8, likelihood=BernoulliLikelihood(), sample_axis=0)
    with torch.no_grad():
        sample = rep.represent(torch.randn(4, 2))

    assert isinstance(sample, TorchBernoulliDistributionSample)
    assert sample.sample_dim == 0
    class_one = sample.tensor.tensor
    assert class_one.shape == (8, 4)
    assert torch.all((class_one >= 0.0) & (class_one <= 1.0))
    assert sample.tensor.probabilities.shape == (8, 4, 2)


def test_bernoulli_representer_with_two_draws_keeps_sample_axis() -> None:
    # Guards against factory heuristics that treat a trailing axis of size <= 2 as a class axis.
    torch.manual_seed(0)
    model = _ScalarSVGP(torch.randn(6, 2)).eval()
    with torch.no_grad():
        sample = representer(model, num_samples=2, likelihood=BernoulliLikelihood()).represent(torch.randn(4, 2))

    assert sample.sample_dim == 1
    assert sample.tensor.tensor.shape == (4, 2)


def test_representer_predict_lists_one_distribution_per_draw(
    softmax_svgp: tuple[_MultitaskSVGP, SoftmaxLikelihood],
) -> None:
    model, likelihood = softmax_svgp
    rep = representer(model, num_samples=3, likelihood=likelihood)
    with torch.no_grad():
        draws = list(rep._predict(torch.randn(4, 2)))  # noqa: SLF001

    assert len(draws) == 3
    assert all(draw.probabilities.shape == (4, NUM_CLASSES) for draw in draws)


def test_representer_rejects_gaussian_likelihood(exact_gp: _ExactGP) -> None:
    with pytest.raises(NotImplementedError, match="GaussianLikelihood"):
        representer(exact_gp, num_samples=4)


def test_representer_requires_likelihood_for_approximate_gp() -> None:
    model = _ScalarSVGP(torch.randn(6, 2)).eval()
    with pytest.raises(TypeError, match="likelihood"):
        representer(model, num_samples=4)


def test_representer_requires_num_samples(exact_gp: _ExactGP) -> None:
    # The explicit class registration must win over the DummyRepresenter that the
    # GaussianDistributionPredictor protocol would otherwise provide.
    with pytest.raises(TypeError, match="num_samples"):
        representer(exact_gp)


def test_conformal_absolute_error_wraps_exact_gp(
    exact_gp: _ExactGP, train_data: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = train_data
    with torch.no_grad():
        calibrated = calibrate(conformal_absolute_error(exact_gp), 0.2, y, x)
        intervals = representer(calibrated).predict(x)

    assert intervals.tensor.shape == (NUM_TRAIN, 2)
    assert torch.all(intervals.tensor[:, 0] <= intervals.tensor[:, 1])
    assert float(coverage(intervals, y)) >= 0.75
