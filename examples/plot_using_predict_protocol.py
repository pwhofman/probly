"""Using the generic `predict()`.

``probly`` defines a small protocol for "predictors" and a generic
:func:`probly.predictor.predict` helper.

This can be useful when you want to dispatch behavior based on the predictor type,
while still supporting the simple ``predictor(x)`` callable pattern.

To make the execution visible in the gallery, this example also produces a small plot
showing the input values and the predicted class labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from probly.predictor import predict


@dataclass(frozen=True)
class ThresholdPredictor:
    """Threshold-based predictor for the gallery demo."""

    threshold: float = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return binary predictions using the configured threshold."""
        return (x > self.threshold).astype(int)


x = np.array([-0.5, 0.0, 0.2, 2.0])
preds = predict(ThresholdPredictor(threshold=0.1), x)

plt.figure(figsize=(4, 2.5))
plt.plot(x, np.zeros_like(x), "o", label="input value")
plt.hlines(0, x.min() - 0.1, x.max() + 0.1, colors="#cccccc", linestyles="--")
plt.scatter(x, preds, color="#d56c6c", marker="s", label="predicted class")
plt.xlabel("Input")
plt.yticks([0, 1], ["class 0", "class 1"])
plt.title("Threshold predictor output")
plt.legend(loc="upper left", frameon=False)
plt.tight_layout()
