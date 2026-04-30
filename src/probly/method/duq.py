"""Deterministic uncertainty quantification method compatibility layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.representation.duq import DUQRepresentation
from probly.transformation.rbf_centroid_head import RBFCentroidHeadPredictor, rbf_centroid_head


@runtime_checkable
class DUQPredictor[**In, Out: DUQRepresentation](RBFCentroidHeadPredictor[In, Out], Protocol):
    """A predictor routed through the DUQ method API."""


duq = DUQPredictor.register_factory(rbf_centroid_head)

__all__ = ["DUQPredictor", "duq"]
