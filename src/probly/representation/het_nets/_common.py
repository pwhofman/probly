"""Shared HetNets representation."""

from __future__ import annotations

from probly.representation.distribution._common import CategoricalDistribution, CategoricalDistributionSample


class HetNetsRepresentation[T: CategoricalDistribution](CategoricalDistributionSample[T]):
    """A sample of categorical distributions produced by a HetNets model.

    HetNets only capture aleatoric uncertainty, so the dispatch on this class
    selects an aleatoric-only quantification.
    """
