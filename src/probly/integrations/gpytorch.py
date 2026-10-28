"""Bindings for GPyTorch models as probly predictors and representers."""

from __future__ import annotations

from typing import Any, cast, override

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood, DirichletClassificationLikelihood, Likelihood, SoftmaxLikelihood
from gpytorch.models import GP
import torch  # noqa: BKN002  # GPyTorch models are torch modules; this module is their torch bridge.

from probly.predictor import predict, predict_raw
from probly.representation.distribution.torch_bernoulli import (
    TorchBernoulliDistributionSample,
    TorchProbabilityBernoulliDistribution,
)
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchLogitCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution, TorchGaussianDistributionSample
from probly.representation.sample import Sample
from probly.representation.sample.torch import TorchSample
from probly.representer._representer import Representer, representer

# Likelihoods whose predictive distribution probly obtains by sampling latent functions.
_CLASSIFICATION_LIKELIHOODS = (BernoulliLikelihood, SoftmaxLikelihood, DirichletClassificationLikelihood)


def _resolve_likelihood(model: GP, likelihood: Likelihood | None) -> Likelihood | None:
    """Return the explicit likelihood, else the one the model owns, else None.

    Exact GPs store their likelihood; approximate GPs do not.
    """
    if likelihood is not None:
        return likelihood
    return getattr(model, "likelihood", None)


def _posterior(model: GP, *args: object, **kwargs: object) -> MultivariateNormal:
    """Call the model and check that it returns a single multivariate normal."""
    posterior = cast("Any", model)(*args, **kwargs)
    if not isinstance(posterior, MultivariateNormal):
        msg = (
            f"{type(model).__name__} returned a {type(posterior).__name__}, but probly supports GPyTorch models "
            "whose call returns a MultivariateNormal. Use the models of an IndependentModelList one by one."
        )
        raise NotImplementedError(msg)
    return posterior


@predict_raw.register(GP)
def gpytorch_predict_raw(
    model: GP, /, *args: object, likelihood: Likelihood | None = None, **kwargs: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the predictive mean and variance of a GPyTorch model.

    The posterior is pushed through ``likelihood`` when one is given, else through the
    likelihood the model owns, which exact GPs always have. Approximate GPs own none, so
    without the keyword their latent posterior is returned, without observation noise.
    This is what GPyTorch users write as ``likelihood(model(x))`` or ``model(x)``.

    The model must be in evaluation mode; GPyTorch raises its own error otherwise. Shapes
    pass through unchanged, so multitask models return ``(n, tasks)`` tensors and batched
    models keep their leading batch dimensions.

    Args:
        model: A GPyTorch model in evaluation mode.
        *args: Forwarded to the model call, typically the input tensor.
        likelihood: Likelihood applied to the posterior. Only likelihoods with a Gaussian
            marginal are supported; use :func:`probly.representer.representer` with
            ``num_samples`` for the others.
        **kwargs: Forwarded to the model call.

    Returns:
        A ``(mean, variance)`` tuple that ``predict`` turns into a ``TorchGaussianDistribution``.

    Raises:
        NotImplementedError: If the likelihood's marginal is not Gaussian.
    """
    posterior = _posterior(model, *args, **kwargs)
    resolved = _resolve_likelihood(model, likelihood)
    if resolved is not None:
        # GPyTorch types ``__call__`` narrowly; cast once.
        posterior = cast("Any", resolved)(posterior)
        if not isinstance(posterior, MultivariateNormal):
            msg = (
                f"predict returns the predictive Gaussian of a GP, but {type(resolved).__name__} has a "
                f"{type(posterior).__name__} marginal. Use representer(model, num_samples=..., likelihood=...) "
                "to sample first-order distributions through the likelihood instead."
            )
            raise NotImplementedError(msg)
    return posterior.mean, posterior.variance


class GpytorchGaussianRepresenter[**In](Representer[Any, In, Any, TorchGaussianDistribution]):
    """Represent a GP by its predictive Gaussian.

    A thin wrapper around :func:`probly.predictor.predict` that binds the likelihood, so
    ``representer(model, likelihood=...)`` and ``representer(model)`` follow the same rules
    as ``predict``.

    Args:
        predictor: A GPyTorch model in evaluation mode.
        likelihood: Likelihood forwarded to ``predict``; None uses the model's own, if any.
    """

    likelihood: Likelihood | None

    def __init__(self, predictor: GP, likelihood: Likelihood | None = None) -> None:
        """Store the model and the likelihood to apply."""
        super().__init__(cast("Any", predictor))
        self.likelihood = likelihood

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> TorchGaussianDistribution:
        """Return the predictive Gaussian for the given inputs."""
        return cast("Any", predict)(self.predictor, *args, likelihood=self.likelihood, **kwargs)


def _softmax_logits(likelihood: SoftmaxLikelihood, draws: torch.Tensor) -> torch.Tensor:
    """Map latent draws through a softmax likelihood, sidestepping its legacy layout check.

    ``SoftmaxLikelihood.forward`` treats inputs whose data axis equals ``num_features`` as legacy
    ``(num_features, num_data)`` tensors and silently transposes them, so one data row is duplicated
    to break the tie and dropped again afterwards.
    """
    num_data = draws.shape[-2]
    padded = num_data == likelihood.num_features
    if padded:
        draws = torch.cat([draws, draws[..., :1, :]], dim=-2)
    logits = cast("Any", likelihood)(draws).logits
    return logits[..., :num_data, :] if padded else logits


class GpytorchSampler[**In, S: Sample](Representer[Any, In, Any, S]):
    """Represent a GP by latent function draws mapped through the likelihood.

    Draws ``num_samples`` latent functions from the GP posterior with reparameterized sampling, maps
    each draw through the likelihood, and stacks the resulting first-order distributions into a sample
    along ``sample_axis``. A Gaussian likelihood gives a
    :class:`~probly.representation.distribution.torch_gaussian.TorchGaussianDistributionSample` whose
    members carry the likelihood noise, so a variance decomposition attributes the posterior spread to
    epistemic and the noise to aleatoric uncertainty. A Bernoulli likelihood gives a
    :class:`~probly.representation.distribution.torch_bernoulli.TorchBernoulliDistributionSample` and
    a softmax likelihood a
    :class:`~probly.representation.distribution.torch_categorical.TorchCategoricalDistributionSample`.
    A Dirichlet classification likelihood gives the same after normalizing the exponentiated draws over
    the class axis, the GPyTorch recipe for exact GP classification. Without a likelihood the draws
    themselves form a :class:`~probly.representation.sample.torch.TorchSample`. Gradients with respect
    to the inputs are preserved.

    Softmax likelihoods expect draws of shape ``(num_samples, n, num_features)``, which is what
    GPyTorch's ``IndependentMultitaskVariationalStrategy`` produces; Dirichlet classification
    likelihoods expect ``(num_samples, num_classes, n)``, the output of a batch-independent exact GP.

    Args:
        predictor: A GPyTorch model in evaluation mode.
        num_samples: Number of latent function draws.
        likelihood: Likelihood applied to the draws; None keeps the latent function values.
        sample_axis: Axis along which draws are stacked in the returned sample. Negative values
            count from the last batch axis, so the default ``-1`` puts draws after the inputs.

    Raises:
        NotImplementedError: When representing, if the likelihood maps draws to a distribution
            probly cannot represent, such as a Student-t or Laplace likelihood.
    """

    num_samples: int
    likelihood: Likelihood | None
    sample_axis: int

    def __init__(
        self,
        predictor: GP,
        num_samples: int,
        likelihood: Likelihood | None = None,
        sample_axis: int = -1,
    ) -> None:
        """Store the model, the likelihood, and the sampling configuration."""
        super().__init__(cast("Any", predictor))
        self.num_samples = num_samples
        self.likelihood = likelihood
        self.sample_axis = sample_axis

    def _stack(self, *parameters: torch.Tensor, num_class_axes: int = 0) -> tuple[list[torch.Tensor], int]:
        """Move the leading draw axis of each parameter tensor to ``sample_axis``."""
        num_batch_axes = parameters[0].ndim - num_class_axes
        sample_dim = self.sample_axis if self.sample_axis >= 0 else num_batch_axes + self.sample_axis
        return [torch.moveaxis(parameter, 0, sample_dim) for parameter in parameters], sample_dim

    def _sample(self, draws: torch.Tensor) -> Sample:
        """Map draws of shape ``(num_samples, *batch)`` through the likelihood and build the sample."""
        likelihood = cast("Any", self.likelihood)
        if likelihood is None:
            (values,), sample_dim = self._stack(draws)
            return TorchSample(values, sample_dim)
        if isinstance(likelihood, BernoulliLikelihood):
            (probabilities,), sample_dim = self._stack(likelihood(draws).probs)
            return TorchBernoulliDistributionSample(TorchProbabilityBernoulliDistribution(probabilities), sample_dim)
        if isinstance(likelihood, SoftmaxLikelihood):
            (logits,), sample_dim = self._stack(_softmax_logits(likelihood, draws), num_class_axes=1)
            return TorchCategoricalDistributionSample(TorchLogitCategoricalDistribution(logits), sample_dim)
        if isinstance(likelihood, DirichletClassificationLikelihood):
            # The draws are (num_samples, num_classes, n): normalize the exponentiated draws over the
            # classes and move that axis last, where categorical distributions keep it.
            probabilities = draws.softmax(dim=-2).movedim(-2, -1)
            (probabilities,), sample_dim = self._stack(probabilities, num_class_axes=1)
            return TorchCategoricalDistributionSample(
                TorchProbabilityCategoricalDistribution(probabilities), sample_dim
            )
        conditional = likelihood(draws)
        base = conditional.base_dist if isinstance(conditional, torch.distributions.Independent) else conditional
        if not isinstance(base, torch.distributions.Normal):
            msg = (
                f"{type(likelihood).__name__} maps latent draws to a {type(conditional).__name__}, which probly "
                "cannot represent. Gaussian, Bernoulli, softmax, and Dirichlet classification likelihoods are "
                "supported."
            )
            raise NotImplementedError(msg)
        (mean, var), sample_dim = self._stack(conditional.mean, conditional.variance)
        return TorchGaussianDistributionSample(TorchGaussianDistribution(mean, var), sample_dim)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> S:
        """Sample latent functions and stack their first-order distributions."""
        posterior = _posterior(cast("Any", self.predictor), *args, **kwargs)
        return cast("Any", self._sample(posterior.rsample(torch.Size([self.num_samples]))))


@representer.register(GP)
def gpytorch_representer(
    predictor: GP,
    num_samples: int | None = None,
    likelihood: Likelihood | None = None,
    sample_axis: int = -1,
) -> Representer[Any, Any, Any, Any]:
    """Select the representer that matches the likelihood and the requested number of draws.

    With ``num_samples`` the GP is represented by a :class:`GpytorchSampler`, which draws latent
    functions and maps them through the likelihood. Without it the GP is represented by its predictive
    Gaussian through a :class:`GpytorchGaussianRepresenter`, which Bernoulli, softmax, and Dirichlet
    classification likelihoods cannot provide. The likelihood is the explicit one, else the model's
    own, else none.

    Args:
        predictor: A GPyTorch model in evaluation mode.
        num_samples: Number of latent draws; required for classification likelihoods.
        likelihood: Likelihood to apply; None uses the model's own, if any.
        sample_axis: Sample axis of the sample built from the draws; ignored for Gaussian outputs.

    Raises:
        TypeError: If ``num_samples`` is missing for a classification likelihood.
    """
    resolved = _resolve_likelihood(predictor, likelihood)
    if num_samples is not None:
        return GpytorchSampler(predictor, num_samples, resolved, sample_axis)
    if isinstance(resolved, _CLASSIFICATION_LIKELIHOODS):
        msg = f"num_samples is required to represent a GP with a {type(resolved).__name__}."
        raise TypeError(msg)
    return GpytorchGaussianRepresenter(predictor, likelihood)
