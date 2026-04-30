"""Torch-side sampler bits.

This module covers two related concerns for torch backends:

1. **Sampling preparation** — :func:`register_forced_train_mode` and the
   built-in registrations enable Monte Carlo techniques like MC Dropout and
   DropConnect by forcing affected layers into train mode during sampling.

2. **Method-specific sampler subclasses** — :class:`LaplaceSampler` hooks
   directly into ``laplace.BaseLaplace.predictive_samples`` so that one bulk
   call replaces ``num_samples`` sequential ``predict`` calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import torch.nn

from probly.layers.torch import DropConnectLinear
from probly.method.laplace import LaplaceGLMPredictor, LaplaceMCPredictor
from probly.representation.distribution import (
    CategoricalDistribution,
    create_categorical_distribution,
)
from probly.representation.sample import Sample
from probly.representer._representer import representer

from ._common import CLEANUP_FUNCS, Sampler, sampling_preparation_traverser

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from flextype.isinstance import LazyType
    from pytraverse import State


def _enforce_train_mode(obj: torch.nn.Module, state: State) -> tuple[torch.nn.Module, State]:
    if not obj.training:
        obj.train()
        state[CLEANUP_FUNCS].add(lambda: obj.train(False))
        return obj, state
    return obj, state


def register_forced_train_mode(cls: LazyType) -> None:
    """Register a class to be forced into train mode during sampling.

    This enables Monte Carlo sampling techniques like MC Dropout :cite:`galDropoutBayesian2016` or DropConnect :cite:
    `mobinyDropConnectEffective2019`.
    """
    sampling_preparation_traverser.register(
        cls,
        _enforce_train_mode,
    )


register_forced_train_mode(
    torch.nn.Dropout
    | torch.nn.Dropout1d
    | torch.nn.Dropout2d
    | torch.nn.Dropout3d
    | torch.nn.AlphaDropout
    | torch.nn.FeatureAlphaDropout
    | DropConnectLinear,
)


@representer.register(LaplaceGLMPredictor)
@representer.register(LaplaceMCPredictor)
class LaplaceSampler[**In, S: Sample](Sampler[In, CategoricalDistribution, S]):
    """Representer that builds a :class:`Sample` of categorical posterior draws.

    Hooks directly into ``laplace.BaseLaplace.predictive_samples`` so that one
    bulk call replaces ``num_samples`` sequential ``predict`` calls. The
    resulting samples are wrapped per-draw in :class:`CategoricalDistribution`
    objects, then aggregated into the chosen :class:`Sample` representation
    via the configured ``sample_factory``.

    Classification only (regression raises ``NotImplementedError`` until a
    torch-backed Gaussian representation is available).
    """

    @override
    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Iterable[CategoricalDistribution]:
        # ``self.predictor`` is typed as the generic ``Predictor`` by Sampler;
        # we know it's a LaplaceMCPredictor here because of the dispatch.
        predictor = cast("Any", self.predictor)
        likelihood = getattr(predictor.la, "likelihood", None)
        if likelihood != "classification":
            msg = f"LaplaceSampler is only implemented for likelihood='classification', got {likelihood!r}."
            raise NotImplementedError(msg)
        if not args:
            msg = "LaplaceSampler.represent expects the input batch as the first argument."
            raise TypeError(msg)
        x = args[0]
        # Strip n_samples from kwargs; it's set from self.num_samples to give
        # the predictor a single source of truth for sample count.
        passthrough = {k: v for k, v in kwargs.items() if k != "n_samples"}
        samples_tensor = predictor.sample(x, n_samples=self.num_samples, **passthrough)
        # samples_tensor: shape (num_samples, batch, num_classes) for classification.
        return [create_categorical_distribution(samples_tensor[i]) for i in range(self.num_samples)]
