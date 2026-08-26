"""Torch SWAG implementation."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from probly.predictor import predict_raw
from probly.representer.sampler._common import CLEANUP_FUNCS, sampling_preparation_traverser

from ._common import collect_swag, swag_generator

if TYPE_CHECKING:
    from pytraverse import State


@torch.no_grad()
def update_swag_stats(
    weights: torch.Tensor,
    mean: torch.Tensor,
    sq_mean: torch.Tensor,
    deviations: torch.Tensor,
    num_collected: int,
) -> None:
    """Update SWAG statistics in place with a new weight snapshot.

    ``deviations`` is used as a ring buffer: the new deviation (relative to the updated running mean) overwrites
    the oldest row, so row order is not chronological. Sampling is unaffected because the low-rank noise is
    isotropic over rows.

    Args:
        weights: Flat weight snapshot of shape ``(d,)``.
        mean: Running first moment of shape ``(d,)``, updated in place.
        sq_mean: Running second moment of shape ``(d,)``, updated in place.
        deviations: Deviation ring buffer of shape ``(max_rank, d)``, updated in place.
        num_collected: Number of snapshots collected before this one.
    """
    mean.lerp_(weights, 1.0 / (num_collected + 1))
    sq_mean.lerp_(weights.square(), 1.0 / (num_collected + 1))
    if deviations.shape[0] > 0:
        deviations[num_collected % deviations.shape[0]].copy_(weights - mean)


@torch.no_grad()
def sample_swag_vector(
    mean: torch.Tensor,
    sq_mean: torch.Tensor,
    deviations: torch.Tensor,
    num_collected: int,
    scale: float,
) -> torch.Tensor:
    """Sample a flat weight vector from the SWAG posterior defined by the given statistics.

    Args:
        mean: Running first moment of shape ``(d,)``.
        sq_mean: Running second moment of shape ``(d,)``.
        deviations: Deviation matrix of shape ``(max_rank, d)``.
        num_collected: Number of collected snapshots.
        scale: Scaling factor for the sampled perturbation.

    Returns:
        A sampled weight vector of shape ``(d,)``.
    """
    # Clamp tiny negative variances from floating-point cancellation; 1e-30 as in the reference implementation.
    variance = torch.clamp(sq_mean - mean.square(), min=1e-30)
    perturbation = variance.sqrt() * torch.randn_like(variance)
    rank = min(num_collected, deviations.shape[0])
    if rank > 1:
        z = torch.randn(rank, dtype=deviations.dtype, device=deviations.device)
        perturbation = perturbation + (z @ deviations[:rank]) / math.sqrt(rank - 1)
    return mean + math.sqrt(scale) * perturbation


def _named_parameter_views(module: nn.Module, vector: torch.Tensor) -> dict[str, torch.Tensor]:
    views = {}
    pointer = 0
    for name, param in module.named_parameters():
        views[name] = vector[pointer : pointer + param.numel()].view_as(param)
        pointer += param.numel()
    return views


class TorchSWAGPredictor(nn.Module):
    """Torch implementation of a SWAG predictor.

    Wraps a copy of the base model and tracks a Gaussian posterior over its flattened weight vector: the running
    mean (the SWA solution), the running second moment, and a low-rank deviation ring buffer holding the last
    ``max_rank`` snapshot deviations from the running mean.

    Train the wrapper exactly like the base model (its parameters are the wrapped model's parameters) and call
    :func:`~probly.method.swag.collect_swag` periodically to record snapshots. During sampling-based prediction
    the forward pass runs the wrapped model with a freshly sampled weight vector via
    ``torch.func.functional_call``, leaving the model's own parameters untouched. The statistics themselves live
    in :func:`update_swag_stats` and :func:`sample_swag_vector`, which can also be used directly on user-held
    tensors.

    Models containing batch normalization require their activation statistics to be recomputed after loading
    sampled weights (e.g. with ``torch.optim.swa_utils.update_bn``) for best results.
    """

    model: nn.Module
    mean: torch.Tensor
    sq_mean: torch.Tensor
    deviations: torch.Tensor
    num_collected: torch.Tensor

    def __init__(self, model: nn.Module, max_rank: int = 20, scale: float = 0.5) -> None:
        """Initialize the SWAG wrapper around a copy of the base model.

        Args:
            model: The base model; it is copied, so the original is not mutated.
            max_rank: Maximum number of rows of the low-rank deviation matrix.
            scale: Default scaling factor for sampled weight perturbations.
        """
        super().__init__()
        self.model = copy.deepcopy(model)
        self.max_rank = max_rank
        self.scale = scale
        self.sampling = False
        weights = parameters_to_vector(self.model.parameters()).detach()
        self.register_buffer("mean", torch.zeros_like(weights))
        self.register_buffer("sq_mean", torch.zeros_like(weights))
        self.register_buffer("deviations", weights.new_zeros((max_rank, weights.numel())))
        self.register_buffer("num_collected", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def collect(self, weights: torch.Tensor | None = None) -> None:
        """Update the SWAG statistics with a weight snapshot.

        Args:
            weights: Flat weight vector to collect. Defaults to the wrapped model's current weights.
        """
        if weights is None:
            weights = parameters_to_vector(self.model.parameters())
        update_swag_stats(weights, self.mean, self.sq_mean, self.deviations, int(self.num_collected))
        self.num_collected += 1

    def _check_collected(self) -> None:
        if int(self.num_collected) == 0:
            msg = "No weight snapshots have been collected yet; call collect_swag during training first."
            raise RuntimeError(msg)

    def sample_weight_vector(self, scale: float | None = None) -> torch.Tensor:
        """Sample a flat weight vector from the SWAG posterior.

        Args:
            scale: Scaling factor for the sampled perturbation. Defaults to the scale given at construction time.

        Returns:
            A sampled weight vector of shape ``(d,)``.
        """
        self._check_collected()
        scale = self.scale if scale is None else scale
        return sample_swag_vector(self.mean, self.sq_mean, self.deviations, int(self.num_collected), scale)

    @torch.no_grad()
    def sample_parameters(self, scale: float | None = None) -> None:
        """Sample a weight vector from the SWAG posterior and load it into the wrapped model.

        Args:
            scale: Scaling factor for the sampled perturbation. Defaults to the scale given at construction time.
        """
        vector_to_parameters(self.sample_weight_vector(scale), self.model.parameters())

    @torch.no_grad()
    def load_mean_parameters(self) -> None:
        """Load the running weight mean (the SWA solution) into the wrapped model."""
        self._check_collected()
        vector_to_parameters(self.mean, self.model.parameters())

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run the wrapped model; in sampling mode with freshly sampled weights, without mutating it."""
        if self.sampling:
            weights = self.sample_weight_vector()
            return torch.func.functional_call(self.model, _named_parameter_views(self.model, weights), args, kwargs)
        return self.model(*args, **kwargs)


@swag_generator.register(nn.Module)
def _torch_swag_generator(
    base: nn.Module,
    max_rank: int,
    scale: float,
    rngs: object,  # noqa: ARG001, sampling uses the global torch generator
) -> TorchSWAGPredictor:
    return TorchSWAGPredictor(base, max_rank, scale)


@predict_raw.register(TorchSWAGPredictor)
def _torch_swag_predict_raw(predictor: TorchSWAGPredictor, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Predict by dispatching on the wrapped model, so its type-specific output handling applies.

    Routing the call through ``predict_raw`` of the wrapped model makes wrappers transparent to
    integrations that adapt model outputs, e.g. the transformers binding that unwraps ``ModelOutput``
    into logits. In sampling mode the sampled weights are loaded into the wrapped model around the
    call and the original weights are restored afterwards.
    """
    if not predictor.sampling:
        return predict_raw(predictor.model, *args, **kwargs)
    saved = parameters_to_vector(predictor.model.parameters()).detach().clone()
    with torch.no_grad():
        vector_to_parameters(predictor.sample_weight_vector(), predictor.model.parameters())
    try:
        return predict_raw(predictor.model, *args, **kwargs)
    finally:
        with torch.no_grad():
            vector_to_parameters(saved, predictor.model.parameters())


@collect_swag.register(TorchSWAGPredictor)
def _torch_collect_swag(predictor: TorchSWAGPredictor) -> None:
    predictor.collect()


def _prepare_swag_sampling(obj: TorchSWAGPredictor, state: State) -> tuple[TorchSWAGPredictor, State]:
    if not obj.sampling:
        obj.sampling = True

        def restore() -> None:
            obj.sampling = False

        state[CLEANUP_FUNCS].add(restore)
    return obj, state


sampling_preparation_traverser.register(TorchSWAGPredictor, _prepare_swag_sampling)
