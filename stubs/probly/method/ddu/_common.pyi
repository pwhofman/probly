"""Shared DDU implementation."""

from __future__ import annotations
import probly
from typing import Literal

from typing import Protocol, runtime_checkable

from lazy_dispatch import lazydispatch
from probly.method.method import predictor_transformation
from probly.predictor import LogitClassifier, Predictor, RepresentationPredictor, predict, predict_raw
from probly.representation.ddu import DDURepresentation, create_ddu_representation
from probly.representation.distribution import create_categorical_distribution_from_logits


@runtime_checkable
class DDUPredictor[**In, Out: DDURepresentation](RepresentationPredictor[In, Out], Protocol):
    """A predictor that applies the credal wrapper representer."""

    encoder: Predictor[In, Out]
    classification_head: Predictor[In, Out]
    density_head: Predictor[In, Out]


@lazydispatch
def ddu_generator[**In, Out: DDURepresentation](
    base: Predictor[In, Out],
    sn_coeff: float,
) -> DDUPredictor[In, Out]:
    """Generate a DDU model from a base model."""
    ...
def ddu[**In, Out: DDURepresentation](base: Predictor[In, Out], sn_coeff: float = 3.0, *, predictor_type: Literal['logit_classifier', 'logit_distribution_predictor'] | type[probly.predictor.LogitDistributionPredictor] | None = None) -> DDUPredictor[In, Out]:
    """Transform a model for Deep Deterministic Uncertainty based on :cite:`mukhotiDeepDeterministicUncertainty2023`.

    Applies spectral normalization to all Conv2d and Linear layers except the
    classification head (the last Linear layer), replaces ReLU and ReLU6
    activations with LeakyReLU(0.01), and replaces stride-1x1 downsampling
    convolutions with AvgPool2d followed by a stride-1 Conv2d.

    The forward pass is unchanged, preserving full training compatibility.

    Args:
        base: Base classification model to be transformed.
        sn_coeff: Lipschitz coefficient for spectral normalization. Weights
            whose spectral norm exceeds this value are rescaled down to it.
            Default is 3.

    Returns:
        The transformed model.
    """
    ...


@predict.register(DDUPredictor)
def predict_ddu_representation[**In, Out: DDURepresentation](
    predictor: DDUPredictor[In, Out],
    *args: In.args,
    **kwargs: In.kwargs,
) -> Out:
    ...
