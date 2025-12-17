"""
=============================
Using the generic `predict()`
=============================

Probly defines a small protocol for "predictors" and a generic
:func:`probly.predictor.predict` helper.

This can be useful when you want to dispatch behavior based on the predictor type,
while still supporting the simple ``predictor(x)`` callable pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from probly.predictor import predict


@dataclass(frozen=True)
class ThresholdPredictor:
    threshold: float = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x > self.threshold).astype(int)


x = np.array([-0.5, 0.0, 0.2, 2.0])
preds = predict(ThresholdPredictor(threshold=0.1), x)
print("preds:", preds)

