"""Shared SWA-Gaussian (SWAG) implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch

from probly.predictor import RandomPredictor
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from flax.nnx.rnglib import Rngs, RngStream

    from probly.predictor import Predictor


@runtime_checkable
class SWAGPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """A predictor with a SWAG posterior over its weights."""


@flexdispatch
def swag_generator[**In, Out](
    base: Predictor[In, Out], max_rank: int, scale: float, rngs: Rngs | RngStream | int
) -> SWAGPredictor[In, Out]:
    """Generate a SWAG predictor from a base predictor."""
    msg = f"No SWAG generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@flexdispatch
def collect_swag(predictor: Predictor) -> None:
    """Collect a snapshot of the current weights into the SWAG statistics.

    Call this from the training loop, typically once per epoch (or every few steps) after the optimizer update,
    once the model has reached a good region of the loss surface. Each call updates the running first and second
    weight moments and the low-rank deviation matrix.

    Args:
        predictor: A predictor created by :func:`swag`.
    """
    msg = f"collect_swag is not implemented for predictors of type {type(predictor)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=True)
@SWAGPredictor.register_factory
def swag[**In, Out](
    base: Predictor[In, Out], max_rank: int = 20, scale: float = 0.5, rngs: Rngs | RngStream | int = 0
) -> SWAGPredictor[In, Out]:
    """Create a SWAG predictor from a base predictor based on :cite:`maddoxSimpleBaseline2019`.

    SWAG (SWA-Gaussian) fits a Gaussian distribution to the weights visited by SGD: the mean is the stochastic
    weight average (SWA) and the covariance is the sum of a diagonal term and a low-rank term formed from the
    last ``max_rank`` weight snapshots. Train the returned predictor as usual and call :func:`collect_swag`
    periodically to record snapshots. At prediction time, drawing repeated predictions (e.g. via
    ``representer(model, num_samples=...)``) samples a fresh weight vector from the fitted Gaussian for every
    forward pass.

    Args:
        base: The base model to be used for SWAG.
        max_rank: Maximum number of columns of the low-rank deviation matrix. Set to 0 for a diagonal-only
            (SWAG-Diag) posterior. Default is 20.
        scale: Scaling factor applied to the sampled weight perturbations. The default of 0.5 is the 1/2
            covariance scaling the paper uses for the full posterior; its diagonal-only variant samples without
            this factor, which corresponds to a scale of 1.0.
        rngs: Optional rngs for the sampling randomness of the flax backend; the torch backend uses the global
            torch generator instead and ignores this. Default is 0.

    Returns:
        The SWAG predictor.
    """
    if max_rank < 0:
        msg = f"The maximum rank must be non-negative, but got {max_rank} instead."
        raise ValueError(msg)
    if scale < 0:
        msg = f"The scale must be non-negative, but got {scale} instead."
        raise ValueError(msg)
    return swag_generator(base, max_rank, scale, rngs)
