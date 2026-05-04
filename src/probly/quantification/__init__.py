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
    Decomposition,
    EpistemicDecomposition,
    SecondOrderEntropyDecomposition,
    SecondOrderZeroOneDecomposition,
    TotalDecomposition,
)
from .measure import (
    conditional_entropy,
    conformal_set_size,
    dempster_shafer_uncertainty,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
    mean_squared_distance_to_scaled_one_hot,
    mutual_information,
    sample_variance,
    spectral_entropy,
    vacuity,
)
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
    "conditional_entropy",
    "conformal_set_size",
    "decompose",
    "dempster_shafer_uncertainty",
    "entropy",
    "entropy_of_expected_predictive_distribution",
    "expected_max_probability_complement",
    "max_disagreement",
    "max_probability_complement_of_expected",
    "mean_squared_distance_to_scaled_one_hot",
    "measure",
    "measure_atomic",
    "mutual_information",
    "notion_registry",
    "quantify",
    "sample_variance",
    "spectral_entropy",
    "vacuity",
]
