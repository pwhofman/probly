"""Flax SWAG implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from flax import nnx
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from probly.representer.sampler._common import CLEANUP_FUNCS, sampling_preparation_traverser

from ._common import collect_swag, swag_generator

if TYPE_CHECKING:
    from pytraverse import State


class SWAGStat(nnx.Variable):
    """Variable type for SWAG statistics, excluded from the trainable parameters."""


def update_swag_stats(
    weights: jax.Array,
    mean: jax.Array,
    sq_mean: jax.Array,
    deviations: jax.Array,
    num_collected: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Update SWAG statistics with a new weight snapshot.

    ``deviations`` is used as a ring buffer: the new deviation (relative to the updated running mean) overwrites
    the oldest row, so row order is not chronological. Sampling is unaffected because the low-rank noise is
    isotropic over rows.

    Args:
        weights: Flat weight snapshot of shape ``(d,)``.
        mean: Running first moment of shape ``(d,)``.
        sq_mean: Running second moment of shape ``(d,)``.
        deviations: Deviation ring buffer of shape ``(max_rank, d)``.
        num_collected: Number of snapshots collected before this one.

    Returns:
        The updated ``(mean, sq_mean, deviations)`` arrays.
    """
    step = 1.0 / (num_collected + 1)
    mean = mean + (weights - mean) * step
    sq_mean = sq_mean + (jnp.square(weights) - sq_mean) * step
    if deviations.shape[0] > 0:
        deviations = deviations.at[num_collected % deviations.shape[0]].set(weights - mean)
    return mean, sq_mean, deviations


def sample_swag_vector(
    key: jax.Array,
    mean: jax.Array,
    sq_mean: jax.Array,
    deviations: jax.Array,
    num_collected: int,
    scale: float,
) -> jax.Array:
    """Sample a flat weight vector from the SWAG posterior defined by the given statistics.

    Args:
        key: JAX random key.
        mean: Running first moment of shape ``(d,)``.
        sq_mean: Running second moment of shape ``(d,)``.
        deviations: Deviation matrix of shape ``(max_rank, d)``.
        num_collected: Number of collected snapshots.
        scale: Scaling factor for the sampled perturbation.

    Returns:
        A sampled weight vector of shape ``(d,)``.
    """
    key_diagonal, key_low_rank = jax.random.split(key)
    # Clamp tiny negative variances from floating-point cancellation; 1e-30 as in the reference implementation.
    variance = jnp.clip(sq_mean - jnp.square(mean), min=1e-30)
    perturbation = jnp.sqrt(variance) * jax.random.normal(key_diagonal, mean.shape, mean.dtype)
    rank = min(num_collected, deviations.shape[0])
    if rank > 1:
        z = jax.random.normal(key_low_rank, (rank,), deviations.dtype)
        perturbation = perturbation + (z @ deviations[:rank]) / math.sqrt(rank - 1)
    return mean + math.sqrt(scale) * perturbation


@swag_generator.register(nnx.Module)
class FlaxSWAGPredictor(nnx.Module):
    """Flax implementation of a SWAG predictor.

    Wraps a clone of the base model and tracks a Gaussian posterior over its flattened parameter vector: the
    running mean (the SWA solution), the running second moment, and a low-rank deviation ring buffer holding the
    last ``max_rank`` snapshot deviations from the running mean. The statistics are stored as :class:`SWAGStat`
    variables, so optimizers targeting ``nnx.Param`` ignore them.

    Train the wrapper exactly like the base model (its parameters are the wrapped model's parameters) and call
    :func:`~probly.method.swag.collect_swag` periodically to record snapshots. During sampling-based prediction
    the wrapped model is rebuilt functionally with a freshly sampled parameter state via ``nnx.merge``, leaving
    the model's own parameters untouched. Sampling randomness is drawn from the ``sample`` stream of the
    wrapper's ``rngs`` attribute.
    """

    def __init__(self, model: nnx.Module, max_rank: int = 20, scale: float = 0.5, rngs: nnx.Rngs | int = 0) -> None:
        """Initialize the SWAG wrapper around a clone of the base model.

        Args:
            model: The base model; it is cloned, so the original is not mutated.
            max_rank: Maximum number of rows of the low-rank deviation matrix.
            scale: Default scaling factor for sampled weight perturbations.
            rngs: Rngs for the sampling randomness.
        """
        self.model = nnx.clone(model)
        self.max_rank = max_rank
        self.scale = scale
        self.sampling = False
        self.rngs = rngs if isinstance(rngs, nnx.Rngs) else nnx.Rngs(rngs)
        weights = self._weight_vector()
        self.mean = SWAGStat(jnp.zeros_like(weights))
        self.sq_mean = SWAGStat(jnp.zeros_like(weights))
        self.deviations = SWAGStat(jnp.zeros((max_rank, weights.size), weights.dtype))
        self.num_collected = SWAGStat(jnp.zeros((), dtype=jnp.int32))

    def _weight_vector(self) -> jax.Array:
        weights, _ = ravel_pytree(nnx.state(self.model, nnx.Param))
        return weights

    def collect(self, weights: jax.Array | None = None) -> None:
        """Update the SWAG statistics with a weight snapshot.

        Args:
            weights: Flat weight vector to collect. Defaults to the wrapped model's current weights.
        """
        if weights is None:
            weights = self._weight_vector()
        num_collected = int(self.num_collected[...])
        mean, sq_mean, deviations = update_swag_stats(
            weights, self.mean[...], self.sq_mean[...], self.deviations[...], num_collected
        )
        self.mean[...] = mean
        self.sq_mean[...] = sq_mean
        self.deviations[...] = deviations
        self.num_collected[...] = num_collected + 1

    def _check_collected(self) -> None:
        if int(self.num_collected[...]) == 0:
            msg = "No weight snapshots have been collected yet; call collect_swag during training first."
            raise RuntimeError(msg)

    def sample_weight_vector(self, scale: float | None = None) -> jax.Array:
        """Sample a flat weight vector from the SWAG posterior.

        Args:
            scale: Scaling factor for the sampled perturbation. Defaults to the scale given at construction time.

        Returns:
            A sampled weight vector of shape ``(d,)``.
        """
        self._check_collected()
        scale = self.scale if scale is None else scale
        return sample_swag_vector(
            self.rngs.sample(),
            self.mean[...],
            self.sq_mean[...],
            self.deviations[...],
            int(self.num_collected[...]),
            scale,
        )

    def _load_vector(self, vector: jax.Array) -> None:
        _, unravel = ravel_pytree(nnx.state(self.model, nnx.Param))
        nnx.update(self.model, unravel(vector))

    def sample_parameters(self, scale: float | None = None) -> None:
        """Sample a weight vector from the SWAG posterior and load it into the wrapped model.

        Args:
            scale: Scaling factor for the sampled perturbation. Defaults to the scale given at construction time.
        """
        self._load_vector(self.sample_weight_vector(scale))

    def load_mean_parameters(self) -> None:
        """Load the running weight mean (the SWA solution) into the wrapped model."""
        self._check_collected()
        self._load_vector(self.mean[...])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run the wrapped model; in sampling mode with freshly sampled weights, without mutating it."""
        if self.sampling:
            graphdef, params, rest = nnx.split(self.model, nnx.Param, ...)
            _, unravel = ravel_pytree(params)
            sampled = unravel(self.sample_weight_vector())
            return nnx.merge(graphdef, sampled, rest)(*args, **kwargs)
        return self.model(*args, **kwargs)


@collect_swag.register(FlaxSWAGPredictor)
def _flax_collect_swag(predictor: FlaxSWAGPredictor) -> None:
    predictor.collect()


def _prepare_swag_sampling(obj: FlaxSWAGPredictor, state: State) -> tuple[FlaxSWAGPredictor, State]:
    if not obj.sampling:
        obj.sampling = True

        def restore() -> None:
            obj.sampling = False

        state[CLEANUP_FUNCS].add(restore)
    return obj, state


sampling_preparation_traverser.register(FlaxSWAGPredictor, _prepare_swag_sampling)
