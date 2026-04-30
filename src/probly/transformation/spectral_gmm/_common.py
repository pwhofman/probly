"""Shared spectral-normalized Gaussian-mixture transformation implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flextype import flexdispatch
from probly.predictor import LogitClassifier, Predictor, RepresentationPredictor
from probly.representation.ddu import DDURepresentation
from probly.transformation.transformation import predictor_transformation


@runtime_checkable
class SpectralGMMPredictor[**In, Out: DDURepresentation](RepresentationPredictor[In, Out], Protocol):
    """A predictor with a spectral-normalized encoder and Gaussian-mixture density head."""

    encoder: Predictor[In, Out]
    classification_head: Predictor[In, Out]
    density_head: Predictor[In, Out]


@flexdispatch
def spectral_gmm_generator[**In, Out: DDURepresentation](
    base: Predictor[In, Out],
    sn_coeff: float,
) -> SpectralGMMPredictor[In, Out]:
    """Generate a spectral-GMM model from a base model."""
    msg = f"No spectral-GMM generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@SpectralGMMPredictor.register_factory
def spectral_gmm[**In, Out: DDURepresentation](
    base: Predictor[In, Out],
    sn_coeff: float = 3.0,
) -> SpectralGMMPredictor[In, Out]:
    """Apply spectral normalization and add a Gaussian-mixture density head.

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
    return spectral_gmm_generator(base, sn_coeff)
