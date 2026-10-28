"""Tests for the optional GPyTorch bindings."""

from __future__ import annotations

import subprocess
import sys
from typing import Any, cast
import warnings

import pytest

pytest.importorskip("torch")
pytest.importorskip("gpytorch")

import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import (
    BernoulliLikelihood,
    DirichletClassificationLikelihood,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
    SoftmaxLikelihood,
    StudentTLikelihood,
)
from gpytorch.models import IndependentModelList
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
import torch

from probly.integrations.gpytorch import GpytorchGaussianRepresenter, GpytorchSampler
from probly.predictor import GaussianDistributionPredictor, predict
from probly.quantification import SecondOrderEntropyDecomposition, SecondOrderVarianceDecomposition
from probly.quantification.measure.variance import variance
from probly.representation.distribution.torch_bernoulli import TorchBernoulliDistributionSample
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.distribution.torch_gaussian import (
    TorchGaussianDistribution,
    TorchGaussianDistributionSample,
)
from probly.representation.sample.torch import TorchSample
from probly.representer import representer

NUM_TRAIN = 24
NUM_CLASSES = 3


class _ExactGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood | FixedNoiseGaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _BatchExactGP(gpytorch.models.ExactGP):
    """One independent GP per batch entry, the GPyTorch recipe for Dirichlet classification."""

    def __init__(
        self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: DirichletClassificationLikelihood, batch: int
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch]))
        self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([batch]))

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


@pytest.fixture
def multitask_gp() -> _MultitaskExactGP:
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, NUM_TRAIN)
    y = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
    return _MultitaskExactGP(x, y, MultitaskGaussianLikelihood(num_tasks=2)).eval()


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


def test_multitask_exact_gp_predict_keeps_task_axis(multitask_gp: _MultitaskExactGP) -> None:
    with torch.no_grad():
        prediction = predict(multitask_gp, torch.tensor([0.1, 0.9]))

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


def test_predict_rejects_model_list(train_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, y = train_data
    models = IndependentModelList(_ExactGP(x, y, GaussianLikelihood()), _ExactGP(x, -y, GaussianLikelihood())).eval()
    with pytest.raises(NotImplementedError, match="MultivariateNormal"):
        predict(models, torch.tensor([0.5]), torch.tensor([0.5]))


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


def test_gaussian_sampler_separates_posterior_spread_from_noise(exact_gp: _ExactGP) -> None:
    x = torch.tensor([0.25, 0.5, 1.5])
    rep = representer(exact_gp, num_samples=4000)
    with torch.no_grad():
        sample = rep.represent(x)
        latent = exact_gp(x)

    assert isinstance(rep, GpytorchSampler)
    assert isinstance(sample, TorchGaussianDistributionSample)
    assert sample.sample_dim == 1
    assert sample.tensor.mean.shape == (3, 4000)
    decomposition = SecondOrderVarianceDecomposition(sample)
    assert torch.allclose(decomposition.aleatoric, cast("Any", exact_gp).likelihood.noise.expand(3))
    assert torch.allclose(decomposition.epistemic, latent.variance, rtol=0.15)


def test_multitask_gaussian_sampler_keeps_task_axis(multitask_gp: _MultitaskExactGP) -> None:
    with torch.no_grad():
        sample = representer(multitask_gp, num_samples=8).represent(torch.tensor([0.1, 0.9]))

    assert isinstance(sample, TorchGaussianDistributionSample)
    assert sample.sample_dim == 2
    assert sample.tensor.mean.shape == (2, 2, 8)
    assert sample.tensor.var.shape == (2, 2, 8)


def test_sampler_without_likelihood_returns_latent_draws() -> None:
    torch.manual_seed(0)
    model = _ScalarSVGP(torch.randn(6, 2)).eval()
    with torch.no_grad():
        sample = representer(model, num_samples=8, sample_axis=0).represent(torch.randn(4, 2))

    assert isinstance(sample, TorchSample)
    assert isinstance(sample.tensor, torch.Tensor)
    assert sample.sample_dim == 0
    assert sample.tensor.shape == (8, 4)


def test_sampler_rejects_unsupported_likelihood(exact_gp: _ExactGP) -> None:
    rep = representer(exact_gp, num_samples=4, likelihood=StudentTLikelihood())
    with pytest.raises(NotImplementedError, match="StudentT"):
        rep.represent(torch.tensor([0.25, 0.75]))


def test_dirichlet_classification_sampler_returns_categorical_sample() -> None:
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, NUM_TRAIN)
    likelihood = DirichletClassificationLikelihood(torch.arange(NUM_TRAIN) % NUM_CLASSES, learn_additional_noise=True)
    model = _BatchExactGP(x, likelihood.transformed_targets, likelihood, NUM_CLASSES).eval()
    with torch.no_grad():
        sample = representer(model, num_samples=8).represent(torch.tensor([0.2, 0.5, 0.8, 0.9]))

    assert isinstance(sample, TorchCategoricalDistributionSample)
    assert sample.sample_dim == 1
    probabilities = sample.tensor.probabilities
    assert probabilities.shape == (4, 8, NUM_CLASSES)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4, 8))


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


def test_softmax_representer_handles_batch_size_equal_to_num_classes(
    softmax_svgp: tuple[_MultitaskSVGP, SoftmaxLikelihood],
) -> None:
    # GPyTorch's softmax likelihood misreads a (num_classes, num_classes) input as its legacy layout
    # and transposes it, emitting a DeprecationWarning. The representer must avoid that path.
    model, likelihood = softmax_svgp
    x = torch.randn(NUM_CLASSES, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with torch.no_grad():
            sample = representer(model, num_samples=8, likelihood=likelihood, sample_axis=0).represent(x)

    probabilities = sample.tensor.probabilities
    assert probabilities.shape == (8, NUM_CLASSES, NUM_CLASSES)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(8, NUM_CLASSES))


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


def test_representer_on_gaussian_gp_passes_through_predict(exact_gp: _ExactGP) -> None:
    x = torch.tensor([0.25, 0.75])
    rep = representer(exact_gp)
    with torch.no_grad():
        represented = rep.predict(x)
        predicted = predict(exact_gp, x)

    assert isinstance(rep, GpytorchGaussianRepresenter)
    assert isinstance(represented, TorchGaussianDistribution)
    assert torch.allclose(represented.mean, predicted.mean)
    assert torch.allclose(represented.var, predicted.var)


def test_predict_applies_explicit_gaussian_likelihood_to_approximate_gp() -> None:
    torch.manual_seed(0)
    model = _ScalarSVGP(torch.randn(6, 2)).eval()
    likelihood = GaussianLikelihood()
    x = torch.randn(4, 2)
    with torch.no_grad():
        latent = predict(model, x)
        predictive = predict(model, x, likelihood=likelihood)
        via_representer = representer(model, likelihood=likelihood).predict(x)

    assert torch.all(predictive.var > latent.var)
    assert torch.allclose(predictive.mean, latent.mean)
    assert torch.allclose(via_representer.var, predictive.var)


def test_predict_rejects_classification_likelihood(softmax_svgp: tuple[_MultitaskSVGP, SoftmaxLikelihood]) -> None:
    model, likelihood = softmax_svgp
    with pytest.raises(NotImplementedError, match="representer"):
        predict(model, torch.randn(4, 2), likelihood=likelihood)


def test_representer_requires_num_samples_for_classification(
    softmax_svgp: tuple[_MultitaskSVGP, SoftmaxLikelihood],
) -> None:
    model, likelihood = softmax_svgp
    with pytest.raises(TypeError, match="num_samples"):
        representer(model, likelihood=likelihood)


class _DeepGPLayer(DeepGPLayer):
    """A layer of the GPyTorch deep GP tutorial with a handful of inducing points."""

    def __init__(self, input_dims: int, output_dims: int | None) -> None:
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
        inducing_points = torch.randn(*batch_shape, 8, input_dims)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(8, batch_shape=batch_shape)
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution)
        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=batch_shape)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class _DeepGP(DeepGP):
    """Two-layer deep GP that owns its likelihood, as in the GPyTorch tutorial."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_layer = _DeepGPLayer(input_dims=2, output_dims=2)
        self.last_layer = _DeepGPLayer(input_dims=2, output_dims=None)
        self.likelihood = GaussianLikelihood()

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return self.last_layer(self.hidden_layer(x))


@pytest.fixture
def deep_gp() -> _DeepGP:
    torch.manual_seed(0)
    model = _DeepGP().eval()
    with torch.no_grad():
        model(torch.zeros(1, 2))  # Initializing the variational distributions on the first call draws random numbers.
    return model


def test_deep_gp_predict_returns_gaussian_batched_over_hidden_draws(deep_gp: _DeepGP) -> None:
    x = torch.rand(5, 2)
    with torch.no_grad():
        prediction = predict(deep_gp, x)

    assert isinstance(prediction, TorchGaussianDistribution)
    assert prediction.mean.shape == (gpytorch.settings.num_likelihood_samples.value(), 5)
    assert prediction.var.shape == prediction.mean.shape


def test_deep_gp_sampler_draws_one_last_layer_function_per_hidden_draw(deep_gp: _DeepGP) -> None:
    x = torch.rand(5, 2)
    rep = representer(deep_gp, num_samples=7)
    with torch.no_grad():
        torch.manual_seed(1)
        sample = rep.represent(x)
        torch.manual_seed(1)
        with gpytorch.settings.num_likelihood_samples(7):
            draws = cast("Any", deep_gp)(x).rsample()

    assert isinstance(rep, GpytorchSampler)
    assert isinstance(sample, TorchGaussianDistributionSample)
    assert sample.sample_dim == 1
    assert draws.shape == (7, 5)
    assert torch.allclose(sample.tensor.mean, draws.movedim(0, -1))
    assert torch.allclose(sample.tensor.var, cast("Any", deep_gp).likelihood.noise.expand(5, 7))


def test_deep_gp_sampler_without_likelihood_returns_latent_draws(deep_gp: _DeepGP) -> None:
    x = torch.rand(3, 2)
    with torch.no_grad():
        torch.manual_seed(1)
        sample = GpytorchSampler(deep_gp, num_samples=4, sample_axis=0).represent(x)
        torch.manual_seed(1)
        with gpytorch.settings.num_likelihood_samples(4):
            draws = cast("Any", deep_gp)(x).rsample()

    assert isinstance(sample, TorchSample)
    assert isinstance(sample.tensor, torch.Tensor)
    assert sample.sample_dim == 0
    assert torch.allclose(sample.tensor, draws)


def test_deep_gp_sampler_maps_draws_through_bernoulli_likelihood(deep_gp: _DeepGP) -> None:
    rep = representer(deep_gp, num_samples=4, likelihood=BernoulliLikelihood())
    with torch.no_grad():
        sample = rep.represent(torch.rand(3, 2))

    assert isinstance(sample, TorchBernoulliDistributionSample)
    assert sample.sample_dim == 1
    assert sample.tensor.tensor.shape == (3, 4)


@pytest.fixture
def fixed_noise_gp(train_data: tuple[torch.Tensor, torch.Tensor]) -> _ExactGP:
    x, y = train_data
    return _ExactGP(x, y, FixedNoiseGaussianLikelihood(noise=torch.full((NUM_TRAIN,), 0.01))).eval()


def test_predict_forwards_likelihood_kwargs(fixed_noise_gp: _ExactGP) -> None:
    x = torch.tensor([0.25, 0.5, 1.5])
    noise = torch.tensor([0.2, 0.3, 0.4])
    with torch.no_grad():
        prediction = predict(fixed_noise_gp, x, likelihood_kwargs={"noise": noise})
        latent = fixed_noise_gp(x)

    assert torch.allclose(prediction.mean, latent.mean)
    assert torch.allclose(prediction.var, latent.variance + noise)


def test_gaussian_representer_forwards_likelihood_kwargs(fixed_noise_gp: _ExactGP) -> None:
    x = torch.tensor([0.25, 0.5, 1.5])
    noise = torch.tensor([0.2, 0.3, 0.4])
    with torch.no_grad():
        represented = representer(fixed_noise_gp, likelihood_kwargs={"noise": noise}).represent(x)
        latent = fixed_noise_gp(x)

    assert torch.allclose(represented.var, latent.variance + noise)


def test_sampler_forwards_likelihood_kwargs(fixed_noise_gp: _ExactGP) -> None:
    x = torch.tensor([0.25, 0.5, 1.5])
    noise = torch.tensor([0.2, 0.3, 0.4])
    with torch.no_grad():
        sample = representer(fixed_noise_gp, num_samples=8, likelihood_kwargs={"noise": noise}).represent(x)

    assert isinstance(sample, TorchGaussianDistributionSample)
    assert sample.tensor.var.shape == (3, 8)
    assert torch.allclose(sample.tensor.var, noise[:, None].expand(3, 8))
