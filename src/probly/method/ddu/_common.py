"""Shared DDU implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flextype import flexdispatch
from probly.method.method import predictor_transformation
from probly.predictor import LogitClassifier, Predictor, RepresentationPredictor
from probly.representation.ddu import DDURepresentation


@runtime_checkable
class DDUPredictor[**In, Out: DDURepresentation](RepresentationPredictor[In, Out], Protocol):
    """A predictor that applies the credal wrapper representer."""

    encoder: Predictor[In, Out]
    classification_head: Predictor[In, Out]
    density_head: Predictor[In, Out]


@flexdispatch
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
