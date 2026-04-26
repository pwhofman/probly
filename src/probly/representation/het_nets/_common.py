"""Shared HetNets representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flextype import flexdispatch
from probly.representation.distribution._common import (
    CategoricalDistribution,
    CategoricalDistributionSample,
)
from probly.utils.iterable import first_element

if TYPE_CHECKING:
    from collections.abc import Iterable


class HetNetsRepresentation[T: CategoricalDistribution](CategoricalDistributionSample[T]):
    """A sample of categorical distributions produced by a HetNets model.

    HetNets only capture aleatoric uncertainty, so the dispatch on this class
    selects an aleatoric-only quantification.
    """

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        """Disable the structural instance hook inherited from ``CategoricalDistributionSample``.

        ``HetNetsRepresentation`` should only match its own subclasses; without this override the
        parent hook would mark every sample of categorical distributions as a HetNets sample,
        causing ``quantify`` to dispatch to ``HetNetsDecomposition`` for all categorical samples.
        """
        del instance
        return NotImplemented


@flexdispatch(dispatch_on=first_element)
def create_het_nets_representation(
    samples: Iterable[Any],
    **_kwargs: Any,  # noqa: ANN401
) -> HetNetsRepresentation:
    msg = f"No HetNets representation factory registered for sample element type {type(first_element(samples))}"
    raise NotImplementedError(msg)
