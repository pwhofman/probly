"""Uncertainty decomposition methods."""

from .decomposition import (
    AdditiveDecomposition,
    AleatoricDecomposition,
    AleatoricEpistemicDecomposition,
    AleatoricEpistemicTotalDecomposition,
    AleatoricTotalDecomposition,
    CachingDecomposition,
    Decomposition,
    EpistemicDecomposition,
    EpistemicTotalDecomposition,
    TotalDecomposition,
)
from .entropy import CredalSetEntropyDecomposition, SecondOrderEntropyDecomposition
from .ordinal import (
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
)
from .scoring_rule import SecondOrderScoringRuleDecomposition
from .spectral import SpectralDecomposition
from .variance import (
    CategoricalVarianceDecomposition,
    SecondOrderVarianceDecomposition,
)
from .wasserstein import SecondOrderWassersteinDecomposition
from .zero_one import SecondOrderZeroOneDecomposition

__all__ = [
    "AdditiveDecomposition",
    "AleatoricDecomposition",
    "AleatoricEpistemicDecomposition",
    "AleatoricEpistemicTotalDecomposition",
    "AleatoricTotalDecomposition",
    "CachingDecomposition",
    "CategoricalVarianceDecomposition",
    "CredalSetEntropyDecomposition",
    "Decomposition",
    "EpistemicDecomposition",
    "EpistemicTotalDecomposition",
    "LabelwiseBinaryEntropyDecomposition",
    "LabelwiseBinaryVarianceDecomposition",
    "OrdinalEntropyDecomposition",
    "OrdinalVarianceDecomposition",
    "SecondOrderEntropyDecomposition",
    "SecondOrderScoringRuleDecomposition",
    "SecondOrderVarianceDecomposition",
    "SecondOrderWassersteinDecomposition",
    "SecondOrderZeroOneDecomposition",
    "SpectralDecomposition",
    "TotalDecomposition",
]
