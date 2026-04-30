"""Deep deterministic uncertainty method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representation.ddu import DDURepresentation
from probly.transformation.spectral_gmm import SpectralGMMPredictor, spectral_gmm


@runtime_checkable
class DDUPredictor[**In, Out: DDURepresentation](SpectralGMMPredictor[In, Out], Protocol):
    """A predictor routed through the DDU method API."""


ddu = DDUPredictor.register_factory(spectral_gmm)

__all__ = ["DDUPredictor", "ddu"]
