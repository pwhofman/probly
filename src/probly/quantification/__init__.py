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
    CategoricalVarianceDecomposition,
    CredalSetEntropyDecomposition,
    Decomposition,
    EpistemicDecomposition,
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
    SecondOrderEntropyDecomposition,
    SecondOrderVarianceDecomposition,
    SecondOrderZeroOneDecomposition,
    TotalDecomposition,
)
from .measure import measure_conformal_set_size, measure_sample_variance
from .notion import (
    AleatoricUncertainty,
    EpistemicUncertainty,
    Notion,
    TotalUncertainty,
    notion_registry,
)

measure = _measure

__all__ = [
    "AdditiveDecomposition",
    "AleatoricDecomposition",
    "AleatoricEpistemicDecomposition",
    "AleatoricEpistemicTotalDecomposition",
    "AleatoricUncertainty",
    "CachingDecomposition",
    "CategoricalVarianceDecomposition",
    "CategoricalVarianceDecomposition",
    "CredalSetEntropyDecomposition",
    "Decomposition",
    "EpistemicDecomposition",
    "EpistemicUncertainty",
    "LabelwiseBinaryEntropyDecomposition",
    "LabelwiseBinaryVarianceDecomposition",
    "Notion",
    "OrdinalEntropyDecomposition",
    "OrdinalVarianceDecomposition",
    "QuantificationResult",
    "Quantifier",
    "SecondOrderEntropyDecomposition",
    "SecondOrderVarianceDecomposition",
    "SecondOrderZeroOneDecomposition",
    "TotalDecomposition",
    "TotalUncertainty",
    "decompose",
    "measure",
    "measure_atomic",
    "measure_conformal_set_size",
    "measure_sample_variance",
    "notion_registry",
    "quantify",
]
