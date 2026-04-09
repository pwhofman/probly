"""Shared DDU implementation."""

from __future__ import annotations

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
    msg = f"No DDU generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@DDUPredictor.register_factory
def ddu[**In, Out: DDURepresentation](
    base: Predictor[In, Out],
    sn_coeff: float = 3.0,
) -> DDUPredictor[In, Out]:
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
    return ddu_generator(base, sn_coeff)


@predict.register(DDUPredictor)
def predict_ddu_representation[**In, Out: DDURepresentation](
    predictor: DDUPredictor[In, Out],
    *args: In.args,
    **kwargs: In.kwargs,
) -> Out:
    raw_prediction = predict_raw(predictor, *args, **kwargs)
    if isinstance(raw_prediction, DDURepresentation):
        return raw_prediction
    if isinstance(raw_prediction, tuple) and len(raw_prediction) == 2:
        logits, densities = raw_prediction
        return create_ddu_representation(
            create_categorical_distribution_from_logits(logits),
            densities,
        )  # ty:ignore[invalid-return-type]

    msg = f"Unexpected prediction type: {type(raw_prediction)}"
    raise ValueError(msg)
