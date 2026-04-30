"""Shared DUQ implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flextype import flexdispatch
from probly.predictor import LogitClassifier, Predictor, RepresentationPredictor
from probly.representation.duq import DUQRepresentation
from probly.transformation.transformation import predictor_transformation


@runtime_checkable
class DUQPredictor[**In, Out: DUQRepresentation](RepresentationPredictor[In, Out], Protocol):
    """A predictor that produces RBF-kernel DUQ uncertainty scores.

    Components:
        encoder: Feature extractor obtained by stripping the original
            classification head from the base model.
        centroid_head: RBF centroid head mapping features to per-class kernel
            values via a learnable per-class projection and EMA-updated class
            centroids.
    """

    encoder: Predictor[In, Out]
    centroid_head: Predictor[In, Out]


@flexdispatch
def duq_generator[**In, Out: DUQRepresentation](
    base: Predictor[In, Out],
    centroid_size: int,
    length_scale: float,
    gamma: float,
) -> DUQPredictor[In, Out]:
    """Generate a DUQ model from a base model."""
    msg = f"No DUQ generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@DUQPredictor.register_factory
def duq[**In, Out: DUQRepresentation](
    base: Predictor[In, Out],
    centroid_size: int = 256,
    length_scale: float = 0.1,
    gamma: float = 0.999,
) -> DUQPredictor[In, Out]:
    r"""Transform a model for Deterministic Uncertainty Quantification :cite:`vanAmersfoortDUQ2020`.

    Replaces the original classification head (the last ``nn.Linear`` layer)
    with an RBF centroid head. For each class :math:`c`, a learnable projection
    :math:`W_c \in \mathbb{R}^{n \times d}` maps the feature vector
    :math:`f_\theta(x) \in \mathbb{R}^d` to an embedding
    :math:`z_c = W_c f_\theta(x)`. The kernel value
    :math:`K_c(x) = \exp\left(-\|z_c - e_c\|^2 / (2 n \sigma^2)\right)` is
    computed against an EMA-updated class centroid :math:`e_c`. The predicted
    class is :math:`\arg\max_c K_c(x)` and the uncertainty score is
    :math:`1 - \max_c K_c(x)`.

    The transformed predictor is intended to be trained from scratch with the
    binary cross-entropy loss on the kernel values and a two-sided gradient
    penalty on the inputs, as in the reference implementation. Class centroids
    must be updated each step via
    :meth:`TorchDUQPredictor.update_centroids`.

    Args:
        base: Base classification model to be transformed.
        centroid_size: Embedding dimension :math:`n` of the per-class projections.
        length_scale: RBF kernel length scale :math:`\\sigma`.
        gamma: Exponential moving-average decay for the class centroids.

    Returns:
        The transformed DUQ predictor.
    """
    return duq_generator(base, centroid_size, length_scale, gamma)
