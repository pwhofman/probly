"""Bindings for GPyTorch models as probly predictors and representers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood, SoftmaxLikelihood
from gpytorch.models import GP
import torch  # noqa: BKN002  # GPyTorch models are torch modules; this module is their torch bridge.

from probly.predictor import predict, predict_raw
from probly.representation.distribution import CategoricalDistribution
from probly.representation.distribution.torch_bernoulli import (
    TorchBernoulliDistributionSample,
    TorchProbabilityBernoulliDistribution,
)
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchLogitCategoricalDistribution,
)
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution
from probly.representation.sample import Sample, create_sample
from probly.representer._representer import Representer, representer
from probly.representer.sampler._common import Sampler

if TYPE_CHECKING:
    from collections.abc import Iterable

_CLASSIFICATION_LIKELIHOODS = (BernoulliLikelihood, SoftmaxLikelihood)


def _resolve_likelihood(model: GP, likelihood: Likelihood | None) -> Likelihood | None:
    """Return the explicit likelihood, else the one the model owns, else None.

    Exact GPs store their likelihood; approximate GPs do not.
    """
    if likelihood is not None:
        return likelihood
    return getattr(model, "likelihood", None)


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
            marginal are supported; use :func:`probly.representer.representer` for
            Bernoulli and softmax likelihoods.
        **kwargs: Forwarded to the model call.

    Returns:
        A ``(mean, variance)`` tuple that ``predict`` turns into a ``TorchGaussianDistribution``.

    Raises:
        NotImplementedError: If the likelihood's marginal is not Gaussian.
    """
    # GPyTorch types ``__call__`` narrowly and ``likelihood`` as optional; cast once.
    gp = cast("Any", model)
    posterior = gp(*args, **kwargs)
    resolved = _resolve_likelihood(model, likelihood)
    if resolved is not None:
        posterior = cast("Any", resolved)(posterior)
        if not isinstance(posterior, MultivariateNormal):
            msg = (
                f"predict supports likelihoods with a Gaussian marginal, but {type(resolved).__name__} "
                f"returned {type(posterior).__name__}. Use representer(model, num_samples=..., "
                "likelihood=...) to obtain class probabilities."
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


class GpytorchClassificationRepresenter[**In, S: Sample](Sampler[In, CategoricalDistribution, S]):
    """Represent a GP classifier by sampling latent functions into a categorical sample.

    Draws ``num_samples`` latent functions from the GP posterior with reparameterized sampling,
    maps each draw through the likelihood, and stacks the resulting class distributions into a
    :class:`~probly.representation.distribution.torch_categorical.TorchCategoricalDistributionSample`
    (softmax likelihood) or a
    :class:`~probly.representation.distribution.torch_bernoulli.TorchBernoulliDistributionSample`
    (Bernoulli likelihood). Gradients with respect to the inputs are preserved.

    Softmax likelihoods expect latent draws of shape ``(num_samples, n, num_features)``, which is
    what GPyTorch's ``IndependentMultitaskVariationalStrategy`` produces. Bernoulli likelihoods
    expect ``(num_samples, n)``.

    Args:
        predictor: A GPyTorch model in evaluation mode.
        num_samples: Number of latent function draws.
        likelihood: A ``SoftmaxLikelihood`` or ``BernoulliLikelihood``.
        sample_axis: Axis along which draws are stacked in the returned sample. Negative values
            count from the last batch axis, so the default ``-1`` puts draws after the inputs.

    Raises:
        NotImplementedError: If the likelihood is neither Bernoulli nor softmax.
    """

    likelihood: Likelihood

    def __init__(
        self,
        predictor: GP,
        num_samples: int,
        likelihood: Likelihood,
        sample_axis: int = -1,
    ) -> None:
        """Validate the likelihood and configure the sampler."""
        if not isinstance(likelihood, _CLASSIFICATION_LIKELIHOODS):
            msg = (
                f"Only BernoulliLikelihood and SoftmaxLikelihood are supported, got {type(likelihood).__name__}. "
                "Gaussian GPs are represented by their predictive distribution; use representer(model)."
            )
            raise NotImplementedError(msg)
        super().__init__(cast("Any", predictor), num_samples, "sequential", create_sample, sample_axis)
        self.likelihood = likelihood

    @property
    def _is_bernoulli(self) -> bool:
        return isinstance(self.likelihood, BernoulliLikelihood)

    def _class_tensor(self, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
        """Draw latent functions and map them through the likelihood.

        Returns class-one probabilities of shape ``(num_samples, *batch)`` for Bernoulli
        likelihoods and logits of shape ``(num_samples, *batch, num_classes)`` otherwise.
        """
        latent = cast("Any", self.predictor)(*args, **kwargs)
        draws = latent.rsample(torch.Size([self.num_samples]))
        likelihood = cast("Any", self.likelihood)
        if self._is_bernoulli:
            return likelihood(draws).probs
        # SoftmaxLikelihood.forward treats inputs whose data axis equals num_features as legacy
        # (num_features x num_data) tensors and silently transposes them. Draws from a multitask
        # posterior are (num_samples, num_data, num_features), so duplicate one data row to break
        # the tie and drop it again afterwards.
        num_data = draws.shape[-2]
        padded = num_data == likelihood.num_features
        if padded:
            draws = torch.cat([draws, draws[..., :1, :]], dim=-2)
        logits = likelihood(draws).logits
        return logits[..., :num_data, :] if padded else logits

    def _distribution(self, tensor: torch.Tensor) -> CategoricalDistribution:
        if self._is_bernoulli:
            return TorchProbabilityBernoulliDistribution(tensor)
        return TorchLogitCategoricalDistribution(tensor)

    @override
    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Iterable[CategoricalDistribution]:
        tensor = self._class_tensor(*args, **kwargs)
        return [self._distribution(tensor[i]) for i in range(self.num_samples)]

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> S:
        """Sample latent functions and return a categorical or Bernoulli sample."""
        tensor = self._class_tensor(*args, **kwargs)
        # The draw axis is first; move it to the requested position among the batch axes.
        # Bernoulli tensors carry no class axis, categorical tensors carry one trailing class axis.
        num_class_axes = 0 if self._is_bernoulli else 1
        target_dim = self.sample_axis if self.sample_axis >= 0 else tensor.ndim - num_class_axes + self.sample_axis
        moved = torch.moveaxis(tensor, 0, target_dim)
        if self._is_bernoulli:
            return TorchBernoulliDistributionSample(  # ty:ignore[invalid-return-type]
                tensor=TorchProbabilityBernoulliDistribution(moved),
                sample_dim=target_dim,
            )
        return TorchCategoricalDistributionSample(  # ty:ignore[invalid-return-type]
            tensor=TorchLogitCategoricalDistribution(moved),
            sample_dim=target_dim,
        )


@representer.register(GP)
def gpytorch_representer(
    predictor: GP,
    num_samples: int | None = None,
    likelihood: Likelihood | None = None,
    sample_axis: int = -1,
) -> Representer[Any, Any, Any, Any]:
    """Select the representer that matches the likelihood.

    The likelihood is the explicit one, else the model's own, else none. Bernoulli and softmax
    likelihoods give a :class:`GpytorchClassificationRepresenter` and require ``num_samples``.
    Anything else, including no likelihood at all, gives a :class:`GpytorchGaussianRepresenter`.

    Args:
        predictor: A GPyTorch model in evaluation mode.
        num_samples: Number of latent draws; required for, and only valid with, classification
            likelihoods.
        likelihood: Likelihood to apply; None uses the model's own, if any.
        sample_axis: Sample axis of the categorical sample; ignored for Gaussian outputs.

    Raises:
        TypeError: If ``num_samples`` is missing for a classification likelihood or given for a
            Gaussian one.
    """
    resolved = _resolve_likelihood(predictor, likelihood)
    if isinstance(resolved, _CLASSIFICATION_LIKELIHOODS):
        if num_samples is None:
            msg = f"num_samples is required to represent a GP with a {type(resolved).__name__}."
            raise TypeError(msg)
        return GpytorchClassificationRepresenter(predictor, num_samples, resolved, sample_axis)
    if num_samples is not None:
        msg = (
            "num_samples only applies to Bernoulli and softmax likelihoods; a GP with a Gaussian "
            "likelihood is represented by its predictive distribution."
        )
        raise TypeError(msg)
    return GpytorchGaussianRepresenter(predictor, likelihood)
