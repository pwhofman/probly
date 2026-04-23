"""River ARF ensemble integration."""

from __future__ import annotations

import numpy as np
from river.forest import ARFClassifier

from probly.predictor import predict_raw
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution

from ._common import EnsembleCategoricalDistributionPredictor

EnsembleCategoricalDistributionPredictor.register(ARFClassifier)


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
        classes = [0]

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
