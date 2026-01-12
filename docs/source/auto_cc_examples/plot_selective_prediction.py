"""
Selective Prediction
====================

This example demonstrates *selective prediction* (abstention).

A model outputs a prediction only when its confidence is above a chosen
threshold; otherwise it rejects the sample.
"""

import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.linspace(0, 10, 100)
y_pred = np.sin(X)
uncertainty = 0.1 + 0.4 * np.abs(np.cos(X))

# Selection threshold
threshold = 0.3
accepted = uncertainty < threshold
rejected = ~accepted

# Plot
plt.figure(figsize=(8, 4))
plt.scatter(X[accepted], y_pred[accepted], label="Accepted predictions")
plt.scatter(X[rejected], y_pred[rejected], label="Rejected predictions", alpha=0.4)

plt.axhline(threshold, color="gray", linestyle="--", label="Uncertainty threshold")
plt.legend()
plt.title("Selective Prediction based on Uncertainty")
plt.xlabel("x")
plt.ylabel("Prediction")
plt.tight_layout()
plt.show()
