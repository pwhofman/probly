"""Uncertainty notion selection with fallback semantics for benchmark scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast
import warnings

from probly.quantification import (
    SecondOrderEntropyDecomposition,
    SecondOrderScoringRuleDecomposition,
    SecondOrderZeroOneDecomposition,
    quantify,
)
from probly.quantification.notion import NotionName, notion_registry
from probly.quantification.scoring_rule import BrierLoss
from probly.representation.distribution import DistributionSample

if TYPE_CHECKING:
    from probly.quantification import Decomposition, QuantificationResult
    from probly.quantification.measure.distribution._common import SecondOrderDistributionLike
    from probly.quantification.notion import Notion
    from probly.representation import Representation

SUPPORTED_DECOMPOSITIONS: tuple[NotionName, ...] = ("aleatoric", "epistemic", "total")

type MeasureName = Literal["default", "entropy", "zero_one", "brier"]
SUPPORTED_MEASURES: tuple[MeasureName, ...] = ("default", "entropy", "zero_one", "brier")


def quantify_with_measure(representation: Representation, measure: MeasureName) -> Decomposition:
    """Quantify a representation with a named second-order uncertainty measure.

    ``"default"`` uses the representation's own :func:`~probly.quantification.quantify`
    dispatch. The other measures build an explicit second-order decomposition on a sample
    of distributions: ``"entropy"`` (:class:`SecondOrderEntropyDecomposition`), ``"zero_one"``
    (:class:`SecondOrderZeroOneDecomposition`) and ``"brier"``
    (:class:`SecondOrderScoringRuleDecomposition` with the Brier score).

    Args:
        representation: Output of a representer, for example a categorical sample.
        measure: One of :data:`SUPPORTED_MEASURES`.

    Returns:
        A decomposition with total, aleatoric and epistemic components (for ``"default"``, the
        representation's own decomposition, which may expose fewer notions).

    Raises:
        ValueError: If ``measure`` is unsupported.
        TypeError: If the representation is not a sample of distributions and ``measure`` is
            not ``"default"``.
    """
    if measure not in SUPPORTED_MEASURES:
        msg = f"Unsupported measure: {measure!r}. Choose from {SUPPORTED_MEASURES}."
        raise ValueError(msg)
    if measure == "default":
        # Every benchmark representation quantifies to a decomposition; see select_uncertainty.
        return cast("Decomposition", quantify(representation))

    if not isinstance(representation, DistributionSample):
        msg = (
            f"Measure {measure!r} needs a sample of distributions, "
            f"got {type(representation).__name__}. Use measure='default' for this method."
        )
        raise TypeError(msg)
    sample = cast("SecondOrderDistributionLike", representation)

    if measure == "entropy":
        return SecondOrderEntropyDecomposition(sample)
    if measure == "zero_one":
        return SecondOrderZeroOneDecomposition(sample)
    return SecondOrderScoringRuleDecomposition(sample, BrierLoss())


_FALLBACK_CHAINS: dict[NotionName, tuple[NotionName, ...]] = {
    "total": ("total", "aleatoric", "epistemic"),
    "aleatoric": ("aleatoric", "total", "epistemic"),
    "epistemic": ("epistemic", "total", "aleatoric"),
}


class UncertaintyFallbackWarning(UserWarning):
    """Warning issued when a fallback uncertainty notion is selected.

    Filter or escalate via the standard ``warnings`` machinery, e.g.
    ``warnings.simplefilter("error", UncertaintyFallbackWarning)`` to turn
    fallbacks into hard errors.
    """


def select_uncertainty(quantification: QuantificationResult, requested: NotionName) -> Notion:
    """Return an uncertainty notion from a decomposition, falling back if missing.

    Some predictors expose decompositions that do not include all of total,
    aleatoric, and epistemic uncertainty (for example, SNGP only exposes
    epistemic uncertainty). When the ``requested`` notion is not a component
    of the decomposition, this function falls back to other notions in a fixed
    order:

    - ``"total"``: total -> aleatoric -> epistemic
    - ``"aleatoric"``: aleatoric -> total -> epistemic
    - ``"epistemic"``: epistemic -> total -> aleatoric

    Args:
        quantification: Result of :func:`probly.quantification.quantify`.
            Must be a :class:`~probly.quantification.Decomposition`; custom
            quantification results that are not decompositions are not
            supported.
        requested: Desired uncertainty notion. Must be one of
            :data:`SUPPORTED_DECOMPOSITIONS`.

    Returns:
        The first notion along the fallback chain that is a component of the
        decomposition.

    Raises:
        ValueError: If ``requested`` is not in :data:`SUPPORTED_DECOMPOSITIONS`,
            or if the decomposition has no notion in the fallback chain.

    Warns:
        UncertaintyFallbackWarning: If a fallback notion is used in place of
            the requested one.
    """
    if requested not in _FALLBACK_CHAINS:
        msg = f"Unsupported decomposition: {requested!r}. Choose from {SUPPORTED_DECOMPOSITIONS}."
        raise ValueError(msg)

    decomposition = cast("Decomposition", quantification)
    chain = _FALLBACK_CHAINS[requested]
    components = decomposition.components
    for name in chain:
        notion_cls = notion_registry[name]
        if notion_cls not in components:
            continue
        if name != requested:
            warnings.warn(
                f"Decomposition {type(decomposition).__name__!r} has no {requested!r} "
                f"component; falling back to {name!r} (chain: {' -> '.join(chain)}).",
                UncertaintyFallbackWarning,
                stacklevel=2,
            )
        return decomposition[name]

    msg = (
        f"Decomposition {type(decomposition).__name__!r} has no notion in the "
        f"fallback chain for {requested!r}: {chain}."
    )
    raise ValueError(msg)
