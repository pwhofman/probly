"""Entropy-based decomposition methods for array-based representations."""

from __future__ import annotations

from probly.quantification.decomposition.entropy._common import EntropyDecomposition


class ArrayEntropyDecomposition(EntropyDecomposition):
    """Base class for entropy-based decomposition methods that operate on array-based representations."""
