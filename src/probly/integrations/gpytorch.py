"""Bindings for GPyTorch models as probly predictors and representers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

from gpytorch.likelihoods import BernoulliLikelihood, Likelihood, SoftmaxLikelihood
from gpytorch.models import ApproximateGP, ExactGP
import torch  # noqa: BKN002  # GPyTorch models are torch modules; this module is their torch bridge.

from probly.predictor import predict_raw
from probly.representation.distribution import CategoricalDistribution
from probly.representation.distribution.torch_bernoulli import (
    TorchBernoulliDistributionSample,
    TorchProbabilityBernoulliDistribution,
)
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchLogitCategoricalDistribution,
)
from probly.representation.sample import Sample, SampleFactory, create_sample
from probly.representer._representer import representer
from probly.representer.sampler._common import Sampler, SamplingStrategy

if TYPE_CHECKING:
    from collections.abc import Iterable


@predict_raw.register(ExactGP)
def gpytorch_exact_predict_raw[**In](
    model: ExactGP, /, *args: In.args, **kwargs: In.kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the predictive mean and variance of an exact GP, observation noise included.

    Calls the model and pushes the latent posterior through ``model.likelihood``, which is what
    GPyTorch users write as ``likelihood(model(x))``. The model must be in evaluation mode;
    GPyTorch raises its own error otherwise. Shapes pass through unchanged, so multitask models
    return ``(n, tasks)`` tensors and batched models keep their leading batch dimensions.

    Args:
        model: An exact GP in evaluation mode.
        *args: Forwarded to the model call, typically the input tensor.
        **kwargs: Forwarded to the model call.

    Returns:
        A ``(mean, variance)`` tuple that ``predict`` turns into a ``TorchGaussianDistribution``.
    """
    # GPyTorch types ``likelihood`` as optional and ``__call__`` narrowly; ExactGP always stores a likelihood.
    predictive = cast("Any", model).likelihood(cast("Any", model)(*args, **kwargs))
    return predictive.mean, predictive.variance


@predict_raw.register(ApproximateGP)
def gpytorch_approximate_predict_raw[**In](
    model: ApproximateGP, /, *args: In.args, **kwargs: In.kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the latent posterior mean and variance of an approximate GP.

    Approximate GPs hold no likelihood, so observation noise is not included. For a classifier
    this is a Gaussian over latent logits; use :func:`probly.representer.representer` with a
    ``likelihood`` argument to obtain class probabilities instead.

    Args:
        model: An approximate (variational) GP in evaluation mode.
        *args: Forwarded to the model call, typically the input tensor.
        **kwargs: Forwarded to the model call.

    Returns:
        A ``(mean, variance)`` tuple that ``predict`` turns into a ``TorchGaussianDistribution``.
    """
    latent = cast("Any", model)(*args, **kwargs)
    return latent.mean, latent.variance


_CLASSIFICATION_LIKELIHOODS = (BernoulliLikelihood, SoftmaxLikelihood)


@representer.register((ExactGP, ApproximateGP))
class GpytorchClassificationRepresenter[**In, S: Sample](Sampler[In, CategoricalDistribution, S]):
    """Representer that samples latent functions of a GP classifier into a categorical sample.

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
        predictor: A GPyTorch ``ExactGP`` or ``ApproximateGP`` in evaluation mode.
        num_samples: Number of latent function draws.
        likelihood: A ``SoftmaxLikelihood`` or ``BernoulliLikelihood``. Defaults to
            ``predictor.likelihood`` when the model owns one, which is the case for exact GPs.
        sampling_strategy: How repeated predictions are computed. Only ``"sequential"`` exists.
        sample_factory: Factory used by the inherited iterable path to build a sample.
        sample_axis: Axis along which draws are stacked in the returned sample. Negative values
            count from the last batch axis, so the default ``-1`` puts draws after the inputs.
    """

    likelihood: Likelihood

    def __init__(
        self,
        predictor: ExactGP | ApproximateGP,
        num_samples: int,
        likelihood: Likelihood | None = None,
        sampling_strategy: SamplingStrategy = "sequential",
        sample_factory: SampleFactory[CategoricalDistribution, S] = create_sample,  # ty:ignore[invalid-parameter-default]
        sample_axis: int = -1,
    ) -> None:
        """Initialize the representer and validate the likelihood."""
        super().__init__(cast("Any", predictor), num_samples, sampling_strategy, sample_factory, sample_axis)
        if likelihood is None:
            likelihood = getattr(predictor, "likelihood", None)
        if likelihood is None:
            msg = (
                "Approximate GPs hold no likelihood; pass likelihood=SoftmaxLikelihood(...) "
                "or likelihood=BernoulliLikelihood() to the representer."
            )
            raise TypeError(msg)
        if not isinstance(likelihood, _CLASSIFICATION_LIKELIHOODS):
            msg = (
                f"Only BernoulliLikelihood and SoftmaxLikelihood are supported, got {type(likelihood).__name__}. "
                "For Gaussian outputs call probly.predictor.predict on the model directly."
            )
            raise NotImplementedError(msg)
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
        conditional = cast("Any", self.likelihood)(draws)
        if self._is_bernoulli:
            return conditional.probs
        return conditional.logits

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
