"""Shared DUQ representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.representation.representation import Representation

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike


@runtime_checkable
class DUQRepresentation(Representation, Protocol):
    r"""Representation of a DUQ model output.

    Holds the per-class RBF kernel values
    :math:`K_c(x) = \exp\left(-\|W_c f_\theta(x) - e_c\|^2 / (2 n \sigma^2)\right)`
    used by Deterministic Uncertainty Quantification :cite:`vanamersfoortDUQ2020`.
    Each value lies in :math:`[0, 1]` and is *not* a probability -- kernel values
    do not sum to one across classes. The predicted class is
    :math:`\arg\max_c K_c(x)` and the uncertainty score is
    :math:`1 - \max_c K_c(x)`.
    """

    @property
    def kernel_values(self) -> ArrayLike:
        """Per-class RBF kernel values, shape ``(..., num_classes)``."""


@flexdispatch
def create_duq_representation(kernel_values: ArrayLike) -> DUQRepresentation:
    """Create a DUQ representation from per-class kernel values."""
    msg = f"No DUQ representation factory registered for kernel values of type {type(kernel_values)}"
    raise NotImplementedError(msg)
