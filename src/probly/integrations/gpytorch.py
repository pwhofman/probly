"""Bindings for GPyTorch models as probly predictors and representers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from gpytorch.models import ApproximateGP, ExactGP

from probly.predictor import predict_raw

if TYPE_CHECKING:
    import torch


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
