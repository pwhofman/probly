"""Torch SWAG implementation."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from probly.representer.sampler._common import CLEANUP_FUNCS, sampling_preparation_traverser

from ._common import collect_swag, swag_generator

if TYPE_CHECKING:
    from pytraverse import State

VAR_CLAMP = 1e-30


@swag_generator.register(nn.Module)
class TorchSWAGPredictor(nn.Module):
    """Torch implementation of a SWAG predictor.

    Wraps a copy of the base model and tracks a Gaussian posterior over its
    flattened weight vector: the running mean (the SWA solution), the running
    second moment, and a low-rank deviation matrix holding the last
    ``max_rank`` snapshot deviations from the running mean.

    Train the wrapper exactly like the base model (its parameters are the
    wrapped model's parameters) and call :func:`~probly.method.swag.collect_swag`
    periodically to record snapshots. During sampling-based prediction the
    wrapper draws a fresh weight vector from the posterior for every forward
    pass and restores the original weights afterwards.

    Note that models containing batch normalization require their activation
    statistics to be recomputed after loading sampled weights (e.g. with
    ``torch.optim.swa_utils.update_bn``) for best results.
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
            max_rank: Maximum number of columns of the low-rank deviation matrix.
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
    def collect(self) -> None:
        """Update the SWAG statistics with the current weights of the wrapped model."""
        weights = parameters_to_vector(self.model.parameters())
        num_collected = int(self.num_collected)
        self.mean.copy_((self.mean * num_collected + weights) / (num_collected + 1))
        self.sq_mean.copy_((self.sq_mean * num_collected + weights.square()) / (num_collected + 1))
        if self.max_rank > 0:
            deviation = weights - self.mean
            if num_collected >= self.max_rank:
                self.deviations.copy_(torch.roll(self.deviations, shifts=-1, dims=0))
            self.deviations[min(num_collected, self.max_rank - 1)].copy_(deviation)
        self.num_collected += 1

    def _check_collected(self) -> None:
        if int(self.num_collected) == 0:
            msg = "No weight snapshots have been collected yet; call collect_swag during training first."
            raise RuntimeError(msg)

    @torch.no_grad()
    def sample_parameters(self, scale: float | None = None) -> None:
        """Sample a weight vector from the SWAG posterior and load it into the wrapped model.

        Args:
            scale: Scaling factor for the sampled perturbation. Defaults to the
                scale given at construction time.
        """
        self._check_collected()
        scale = self.scale if scale is None else scale
        variance = torch.clamp(self.sq_mean - self.mean.square(), min=VAR_CLAMP)
        perturbation = variance.sqrt() * torch.randn_like(variance)
        rank = min(int(self.num_collected), self.max_rank)
        if rank > 1:
            z = torch.randn(rank, dtype=self.deviations.dtype, device=self.deviations.device)
            perturbation = perturbation + (z @ self.deviations[:rank]) / math.sqrt(rank - 1)
        vector_to_parameters(self.mean + math.sqrt(scale) * perturbation, self.model.parameters())

    @torch.no_grad()
    def load_mean_parameters(self) -> None:
        """Load the running weight mean (the SWA solution) into the wrapped model."""
        self._check_collected()
        vector_to_parameters(self.mean, self.model.parameters())

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run the wrapped model, resampling its weights first when in sampling mode."""
        if self.sampling:
            self.sample_parameters()
        return self.model(*args, **kwargs)


@collect_swag.register(TorchSWAGPredictor)
def _torch_collect_swag(predictor: TorchSWAGPredictor) -> None:
    predictor.collect()


def _prepare_swag_sampling(obj: TorchSWAGPredictor, state: State) -> tuple[TorchSWAGPredictor, State]:
    if not obj.sampling:
        saved = parameters_to_vector(obj.model.parameters()).detach().clone()

        def restore() -> None:
            obj.sampling = False
            with torch.no_grad():
                vector_to_parameters(saved, obj.model.parameters())

        obj.sampling = True
        state[CLEANUP_FUNCS].add(restore)
    return obj, state


sampling_preparation_traverser.register(TorchSWAGPredictor, _prepare_swag_sampling)
