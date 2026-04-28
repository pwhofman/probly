"""Quantification of DUQ representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from flextype import flexdispatch
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import CachingDecomposition, TotalDecomposition
from probly.representation.duq import DUQRepresentation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from probly.representation.array_like import ArrayLike


@flexdispatch
def duq_uncertainty(kernel_values: Iterable) -> ArrayLike:
    r"""Compute the DUQ uncertainty score :math:`1 - \max_c K_c(x)`.

    The DUQ kernel value :math:`\\max_c K_c(x)` is a confidence score in
    :math:`[0, 1]`; its complement is used as a single-number uncertainty
    measure :cite:`vanamersfoortDUQ2020`. DUQ does not provide an explicit
    aleatoric/epistemic decomposition, so a single total-uncertainty score is
    returned.

    Args:
        kernel_values: kernel values of shape ``(..., C)``.

    Returns:
        Per-sample uncertainty scores of shape ``(...,)`` (the trailing class
        axis of ``kernel_values`` is reduced via max).
    """
    msg = f"DUQ uncertainty is not implemented for kernel values of type {type(kernel_values)}"
    raise NotImplementedError(msg)


@decompose.register(DUQRepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class DUQDecomposition[T](CachingDecomposition, TotalDecomposition[T]):
    """Base class for entropy-based decomposition methods."""

    representation: DUQRepresentation

    @override
    @property
    def _total(self) -> T:
        """The total uncertainty of the decomposition."""
        return duq_uncertainty(self.representation.kernel_values)  # ty:ignore[invalid-return-type]
