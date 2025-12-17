"""
Epistemic vs. Aleatoric Uncertainty
==================================

This example illustrates the conceptual difference between epistemic and
aleatoric uncertainty using a simple regression toy problem.

Aleatoric uncertainty is modeled as noise in the data.
Epistemic uncertainty is illustrated by a lack of data in certain regions
of the input space.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# True function
x = np.linspace(0, 6, 200)
y_true = np.sin(x)

# Aleatoric uncertainty: noisy observations
noise = rng.normal(0, 0.3, size=len(x))
y_noisy = y_true + noise

# Epistemic uncertainty: missing data region
train_mask = x < 4.0
x_train = x[train_mask]
y_train = y_noisy[train_mask]

plt.figure(figsize=(6, 4))
plt.scatter(x_train, y_train, s=15, label="Observed data (with noise)")
plt.plot(x, y_true, color="black", label="True function")
plt.axvspan(4.0, 6.0, alpha=0.2, label="Epistemic uncertainty region")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Aleatoric vs. Epistemic Uncertainty")
plt.legend()
plt.tight_layout()
plt.show()
