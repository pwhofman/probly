"""Shared HetNets representer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from probly.method.het_nets import HetNetsPredictor
from probly.representation.het_nets import HetNetsRepresentation
from probly.representer._representer import representer
from probly.representer.sampler import Sampler

if TYPE_CHECKING:
    from collections.abc import Iterable


@representer.register(HetNetsPredictor)
class HetNetsRepresenter[**In, Out](Sampler[In, Out, HetNetsRepresentation]):
    """A representer that draws samples from a HetNets predictor.

    Each call to the underlying predictor produces a single categorical distribution
    drawn by sampling once from the heteroscedastic latent utility model. Repeated
    calls form a Monte Carlo sample of categorical distributions.
    """

    @override
    def _create_sample(self, predictions: Iterable[Out]) -> HetNetsRepresentation:
        return HetNetsRepresentation.register_instance(super()._create_sample(predictions))
