"""Shared SWA-Gaussian (SWAG) implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch

from probly.predictor import RandomPredictor
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from probly.predictor import Predictor


@runtime_checkable
class SWAGPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
    """A predictor with a SWAG posterior over its weights."""


@flexdispatch
def swag_generator[T: Predictor](base: T, max_rank: int, scale: float) -> T:
    """Generate a SWAG predictor from a base predictor."""
    msg = f"No SWAG generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@flexdispatch
def collect_swag(predictor: Predictor) -> None:
    """Collect a snapshot of the current weights into the SWAG statistics.

    Call this from the training loop, typically once per epoch (or every few
    steps) after the optimizer update, once the model has reached a good
    region of the loss surface. Each call updates the running first and second
    weight moments and appends a column to the low-rank deviation matrix.

    Args:
        predictor: A predictor created by :func:`swag`.
    """
    msg = f"collect_swag is not implemented for predictors of type {type(predictor)}"
    raise NotImplementedError(msg)


@flexdispatch
def swag_snapshot_generator[T: Predictor](base: T, snapshots: Iterable, max_rank: int, scale: float) -> T:
    """Generate a SWAG predictor from a base predictor and saved weight snapshots."""
    msg = f"No SWAG snapshot generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


def _validate_swag_args(max_rank: int, scale: float) -> None:
    if max_rank < 0:
        msg = f"The maximum rank must be non-negative, but got {max_rank} instead."
        raise ValueError(msg)
    if scale < 0:
        msg = f"The scale must be non-negative, but got {scale} instead."
        raise ValueError(msg)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=True)
@SWAGPredictor.register_factory
def swag[T: Predictor](base: T, max_rank: int = 20, scale: float = 0.5) -> T:
    """Create a SWAG predictor from a base predictor based on :cite:`maddoxSimpleBaseline2019`.

    SWAG (SWA-Gaussian) fits a Gaussian distribution to the weights visited by
    SGD: the mean is the stochastic weight average (SWA) and the covariance is
    the sum of a diagonal term and a low-rank term formed from the last
    ``max_rank`` weight snapshots. Train the returned predictor as usual and
    call :func:`collect_swag` periodically to record snapshots. At prediction
    time, drawing repeated predictions (e.g. via
    ``representer(model, num_samples=...)``) samples a fresh weight vector from
    the fitted Gaussian for every forward pass.

    Args:
        base: The base model to be used for SWAG.
        max_rank: Maximum number of columns of the low-rank deviation matrix.
            Set to 0 to use a diagonal-only (SWAG-Diag) posterior. Default is 20.
        scale: Scaling factor applied to the sampled weight perturbations. The
            default of 0.5 corresponds to the 1/2 covariance factors used in the
            paper.

    Returns:
        The SWAG predictor.
    """
    _validate_swag_args(max_rank, scale)
    return swag_generator(base, max_rank, scale)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=True)
@SWAGPredictor.register_factory
def swag_from_snapshots[T: Predictor](base: T, snapshots: Iterable, max_rank: int = 20, scale: float = 0.5) -> T:
    """Create a SWAG predictor from saved weight snapshots based on :cite:`maddoxSimpleBaseline2019`.

    Fits the SWAG posterior post hoc by replaying previously saved snapshots,
    e.g. checkpoints collected during an earlier training run. The wrapped
    model keeps the weights of ``base``; only the SWAG statistics are built
    from the snapshots.

    Args:
        base: The base model to be used for SWAG.
        snapshots: Weight snapshots, each either a flat weight vector or a
            model of the same architecture as ``base``.
        max_rank: Maximum number of columns of the low-rank deviation matrix.
            Set to 0 to use a diagonal-only (SWAG-Diag) posterior. Default is 20.
        scale: Scaling factor applied to the sampled weight perturbations.
            Default is 0.5.

    Returns:
        The SWAG predictor.
    """
    _validate_swag_args(max_rank, scale)
    return swag_snapshot_generator(base, snapshots, max_rank, scale)
