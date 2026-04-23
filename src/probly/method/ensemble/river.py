"""River ARF ensemble integration."""

from __future__ import annotations

import numpy as np
from river.forest import ARFClassifier, ARFRegressor

from probly.predictor import predict_raw
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution
from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution

from ._common import EnsembleCategoricalDistributionPredictor, EnsemblePredictor

EnsembleCategoricalDistributionPredictor.register(ARFClassifier)
EnsemblePredictor.register(ARFRegressor)


@predict_raw.register(ARFClassifier)
def predict_arf_ensemble(arf: ARFClassifier, x: dict[str, float]) -> list[ArrayCategoricalDistribution]:
    """Extract aligned per-tree categorical distributions from an ARF."""
    per_tree_dicts = [m.predict_proba_one(x) for m in arf.models]

    # Infer common class order from all trees and the ensemble itself
    seen: set = set()
    for pt in per_tree_dicts:
        seen.update(pt)
    seen.update(arf.predict_proba_one(x))
    classes = sorted(seen, key=lambda c: (not isinstance(c, (int, float)), str(c)))
    if not classes:
        classes = [0, 1]

    class_idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)

    result = []
    for pt in per_tree_dicts:
        probs = np.zeros(k, dtype=np.float64)
        total = sum(pt.values())
        if total <= 0:
            probs[:] = 1.0 / k
        else:
            for cls, p in pt.items():
                probs[class_idx[cls]] = p / total
        result.append(ArrayCategoricalDistribution(unnormalized_probabilities=probs))
    return result


_TREE_VAR = np.array([1e-8])
"""Near-zero variance for individual tree predictions (no noise model)."""


@predict_raw.register(ARFRegressor)
def predict_arf_regressor_ensemble(arf: ARFRegressor, x: dict[str, float]) -> list[ArrayGaussianDistribution]:
    """Extract per-tree Gaussian distributions from an ARF regressor.

    Each tree produces a point prediction wrapped as a near-degenerate Gaussian.
    Epistemic uncertainty arises from disagreement across trees.
    """
    return [ArrayGaussianDistribution(mean=np.array([float(m.predict_one(x))]), var=_TREE_VAR) for m in arf.models]
