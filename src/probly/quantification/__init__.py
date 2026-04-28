"""Quantification methods for uncertainty."""

from ._quantification import (
    QuantificationResult,
    Quantifier,
    decompose,
    measure as _measure,  # to avoid name clash with later .measure import
    measure_atomic,
    quantify,
)
from .decomposition import (
    AdditiveDecomposition,
    AleatoricDecomposition,
    AleatoricEpistemicDecomposition,
    AleatoricEpistemicTotalDecomposition,
    CachingDecomposition,
    CredalSetEntropyDecomposition,
    DDUDensityDecomposition,
    Decomposition,
    EpistemicDecomposition,
    SecondOrderEntropyDecomposition,
    SecondOrderZeroOneDecomposition,
    TotalDecomposition,
)
from .measure import measure_sample_variance
from .notion import (
    AleatoricUncertainty,
    EpistemicUncertainty,
    Notion,
    TotalUncertainty,
)

measure = _measure

__all__ = [
    "AdditiveDecomposition",
    "AleatoricDecomposition",
    "AleatoricEpistemicDecomposition",
    "AleatoricEpistemicTotalDecomposition",
    "AleatoricUncertainty",
    "CachingDecomposition",
    "CredalSetEntropyDecomposition",
    "DDUDensityDecomposition",
    "Decomposition",
    "EpistemicDecomposition",
    "EpistemicUncertainty",
    "Notion",
    "QuantificationResult",
    "Quantifier",
    "SecondOrderEntropyDecomposition",
    "SecondOrderZeroOneDecomposition",
    "TotalDecomposition",
    "TotalUncertainty",
    "decompose",
    "measure",
    "measure_atomic",
    "measure_sample_variance",
    "notion_registry",
    "quantify",
]
