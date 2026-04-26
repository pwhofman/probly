"""Quantification of DUQ representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flextype import flexdispatch
from probly.quantification._quantification import quantify
from probly.representation.duq import DUQRepresentation

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike


@flexdispatch
def duq_uncertainty(representation: DUQRepresentation) -> ArrayLike:
    r"""Compute the DUQ uncertainty score :math:`1 - \max_c K_c(x)`.

    The DUQ kernel value :math:`\\max_c K_c(x)` is a confidence score in
    :math:`[0, 1]`; its complement is used as a single-number uncertainty
    measure :cite:`vanamersfoortDUQ2020`. DUQ does not provide an explicit
    aleatoric/epistemic decomposition, so a single total-uncertainty score is
    returned.

    Args:
        representation: DUQ representation produced by a DUQ predictor.

    Returns:
        Per-sample uncertainty scores of shape ``(...,)`` (the trailing class
        axis of ``kernel_values`` is reduced via max).
    """
    msg = f"DUQ uncertainty is not implemented for representations of type {type(representation)}"
    raise NotImplementedError(msg)


@quantify.register(DUQRepresentation)
def _quantify_duq(representation: DUQRepresentation) -> ArrayLike:
    """Quantify the uncertainty of a DUQ representation."""
    return duq_uncertainty(representation)
