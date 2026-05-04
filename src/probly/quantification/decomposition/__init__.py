"""Uncertainty decomposition methods."""

from . import spectral as spectral
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
from .ordinal import OrdinalEntropyDecomposition, OrdinalVarianceDecomposition
from .variance import OrdinalIntegerVarianceDecomposition, SecondOrderVarianceDecomposition
from .zero_one import SecondOrderZeroOneDecomposition


def __getattr__(name: str) -> object:
    if name in {"SpectralDecomposition", "spectral_decomposition"}:
        return getattr(spectral, name)
    raise AttributeError(name)


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
    "OrdinalEntropyDecomposition",
    "OrdinalIntegerVarianceDecomposition",
    "OrdinalVarianceDecomposition",
    "SecondOrderEntropyDecomposition",
    "SecondOrderVarianceDecomposition",
    "SecondOrderZeroOneDecomposition",
    "SpectralDecomposition",
    "TotalDecomposition",
    "spectral_decomposition",
]
