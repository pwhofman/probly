"""HetNets method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representation.distribution import CategoricalDistribution
from probly.transformation.heteroscedastic_classification import (
    HeteroscedasticClassificationPredictor,
    heteroscedastic_classification,
    heteroscedastic_classification_traverser,
)


@runtime_checkable
class HetNetsPredictor[**In, Out: CategoricalDistribution](HeteroscedasticClassificationPredictor[In, Out], Protocol):
    """A predictor routed through the HetNets representer."""


het_nets = HetNetsPredictor.register_factory(heteroscedastic_classification)
het_nets_traverser = heteroscedastic_classification_traverser

__all__ = ["HetNetsPredictor", "het_nets", "het_nets_traverser"]
