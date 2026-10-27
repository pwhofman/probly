"""Per-method decision logic for producing class probabilities.

Default: ``representer`` -> ``categorical_from_mean``.
Methods that need custom decision logic (e.g. using the base model or
the MLE instead of the full uncertainty representation) register here.

Follows the same ``flexdispatch`` pattern as :func:`train_model`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flextype import flexdispatch
import torch

from probly.decider import categorical_from_mean
from probly.method.batchensemble import BatchEnsemblePredictor
from probly.method.credal_relative_likelihood import CredalRelativeLikelihoodPredictor
from probly.method.dare import DarePredictor
from probly.method.ddu import DDUPredictor
from probly.method.efficient_credal_prediction import EfficientCredalPredictor
from probly.method.subensemble import SubensemblePredictor
from probly.predictor import predict
from probly.representation.distribution import (
    create_categorical_distribution,
    create_categorical_distribution_from_logits,
)
from probly.representer import representer

if TYPE_CHECKING:
    from torch import nn

    from probly.representation.distribution import CategoricalDistribution


def dare_probabilities(outputs: torch.Tensor) -> torch.Tensor:
    """Class probabilities of DARE head outputs: clipped at zero and normalized per instance.

    The heads regress onto one-hot targets scaled by a constant with a linear output, so their outputs
    estimate the class posterior up to that scale; the paper reports accuracy and AUROC only and defines
    no predictive distribution. An instance whose outputs are all non-positive gets the uniform distribution.

    Args:
        outputs: Head outputs of shape ``(..., num_classes)``.

    Returns:
        Probabilities of the same shape.
    """
    clipped = outputs.float().clamp(min=0.0)
    total = clipped.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(clipped, 1.0 / clipped.shape[-1])
    return torch.where(total > 0, clipped / total.clamp(min=torch.finfo(torch.float32).eps), uniform)


@flexdispatch
def decide(
    model: nn.Module,
    x: torch.Tensor,
    rep_kwargs: dict[str, Any] | None = None,
) -> CategoricalDistribution:
    """Produce a categorical decision from a model.

    Default: ``representer`` -> ``categorical_from_mean``.
    Register custom handlers for methods that need different decision logic.

    Args:
        model: The (possibly wrapped) model to make a decision with.
        x: Input tensor.
        rep_kwargs: Representer parameters (e.g. ``num_samples``).

    Returns:
        A categorical distribution representing the class-probability decision.
    """
    return categorical_from_mean(representer(model, **(rep_kwargs or {})).represent(x))


@decide.register(EfficientCredalPredictor)
def _decide_efficient_credal(
    model: EfficientCredalPredictor,
    x: torch.Tensor,
    rep_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> CategoricalDistribution:
    """Use the base model's prediction for EfficientCredalPrediction."""
    return predict(model.predictor, x)


@decide.register(CredalRelativeLikelihoodPredictor)
def _decide_credal_relative_likelihood(
    model: CredalRelativeLikelihoodPredictor,
    x: torch.Tensor,
    rep_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> CategoricalDistribution:
    """Use the MLE (first ensemble member) for CredalRelativeLikelihood."""
    return predict(model[0], x)


@decide.register(DDUPredictor)
def _decide_ddu(
    model: DDUPredictor,
    x: torch.Tensor,
    rep_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> CategoricalDistribution:
    """Semantically the same as the default, but skips the density head computation."""
    features = model.encoder(x)
    logits = model.classification_head(features)
    return create_categorical_distribution_from_logits(logits)


@decide.register(SubensemblePredictor)
def _decide_subensemble(
    model: SubensemblePredictor,
    x: torch.Tensor,
    rep_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> CategoricalDistribution:
    """Average softmax probabilities across all subensemble members."""
    member_dists = [create_categorical_distribution_from_logits(predict(member, x)) for member in model]
    avg_probs = sum(d.probabilities for d in member_dists) / len(member_dists)
    return create_categorical_distribution(avg_probs)


@decide.register(DarePredictor)
def _decide_dare(
    model: DarePredictor,
    x: torch.Tensor,
    rep_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> CategoricalDistribution:
    """Average the clipped, normalized head outputs across all DARE members."""
    member_probs = [dare_probabilities(member(x)) for member in model]
    return create_categorical_distribution(torch.stack(member_probs).mean(dim=0))


@decide.register(BatchEnsemblePredictor)
def _decide_batchensemble(
    model: BatchEnsemblePredictor,
    x: torch.Tensor,
    rep_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> CategoricalDistribution:
    """Average softmax probabilities across all batchensemble members."""
    sample = predict(model, x)
    member_dists = [create_categorical_distribution_from_logits(logits) for logits in sample.samples]
    avg_probs = sum(d.probabilities for d in member_dists) / len(member_dists)
    return create_categorical_distribution(avg_probs)
