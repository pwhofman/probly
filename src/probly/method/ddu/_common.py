"""Shared DDU implementation."""

from __future__ import annotations

from typing import Protocol

from lazy_dispatch import lazydispatch
from probly.predictor import Predictor


class DDUPredictor[**In, Out](Predictor[In, Out], Protocol):
    """A predictor that applies the credal wrapper representer."""


@lazydispatch
def ddu_generator[**In, Out](base: Predictor[In, Out], sn_coeff: float) -> DDUPredictor[In, Out]:
    """Generate a DDU model from a base model."""
    msg = f"No DDU generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@DDUPredictor.register_factory
def ddu[**In, Out](base: Predictor[In, Out], sn_coeff: float = 3.0) -> DDUPredictor[In, Out]:
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
