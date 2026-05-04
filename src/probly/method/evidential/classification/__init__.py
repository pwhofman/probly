"""Evidential classification method compatibility layer."""

from __future__ import annotations

from probly.transformation.dirichlet_clipped_exp_one_activation import register

from ._common import (
    EvidentialClassificationDecomposition,
    EvidentialClassificationPredictor,
    evidential_classification,
)

__all__ = [
    "EvidentialClassificationDecomposition",
    "EvidentialClassificationPredictor",
    "evidential_classification",
    "register",
]
