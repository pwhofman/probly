"""Epistemic vs Aleatoric Uncertainty.

This example illustrates the difference between epistemic and aleatoric
uncertainty in probabilistic models.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Random seed
rng = np.random.RandomState(0)

# Data
X = np.linspace(0, 10, 100)
y_true = np.sin(X)

# Aleatoric uncertainty (noise)
noise = rng.normal(scale=0.2, size=len(X))
y_aleatoric = y_true + noise

# Epistemic uncertainty (model variance simulation)
epistemic_samples = [y_true + rng.normal(scale=0.1, size=len(X)) for _ in range(10)]

# Plot
plt.figure(figsize=(8, 4))
plt.plot(X, y_true, label="True function", linewidth=2)
plt.scatter(X, y_aleatoric, label="Aleatoric noise", alpha=0.5)

for ys in epistemic_samples:
    plt.plot(X, ys, color="gray", alpha=0.3)

plt.legend()
plt.title("Epistemic vs. Aleatoric Uncertainty")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
