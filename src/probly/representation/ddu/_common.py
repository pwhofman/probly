"""Shared DDU representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from lazy_dispatch import lazydispatch
from probly.representation.representation import Representation

if TYPE_CHECKING:
    from probly.representation.distribution._common import CategoricalDistribution


class DDURepresentation(Representation, Protocol):
    """Representation of a DDU model output.

    Holds the two quantities needed for uncertainty quantification:
    softmax probabilities (aleatoric) and feature vectors (epistemic).
    """

    softmax: CategoricalDistribution
    features: Any


@lazydispatch
def create_ddu_representation(softmax: CategoricalDistribution, features: Any) -> DDURepresentation:  # noqa: ANN401
    """Create a DDU representation from a softmax distribution and feature vectors."""
    msg = f"No DDU representation factory registered for softmax type {type(softmax)}"
    raise NotImplementedError(msg)
