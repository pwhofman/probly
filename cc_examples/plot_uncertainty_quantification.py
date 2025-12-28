<<<<<<< Updated upstream
"""
Uncertainty Quantification
=================================

This is a minimal Sphinx-Gallery example.
"""

import numpy as np
import matplotlib.pyplot as plt


# %%
# Create a synthetic regression dataset
# -------------------------------------

rng = np.random.default_rng(123)

X = np.linspace(-4, 4, 250)
true_function = 0.5 * np.sin(X) + 0.1 * X

# Observation noise (aleatoric)
obs_std = 0.2
y = true_function + rng.normal(0, obs_std, size=len(X))


# %%
# Simulate a predictive distribution
# ----------------------------------
#
# We simulate model uncertainty by drawing multiple "model predictions".
# Think of these as predictions from an ensemble of models.

n_samples = 50
pred_samples = []

for _ in range(n_samples):
    # Simulated epistemic uncertainty (model variation)
    model_variation = rng.normal(0, 0.12, size=len(X))
    pred_samples.append(true_function + model_variation)

pred_samples = np.asarray(pred_samples)

pred_mean = pred_samples.mean(axis=0)
pred_std = pred_samples.std(axis=0)


# %%
# Plot uncertainty quantification
# -------------------------------

plt.figure(figsize=(8, 4))

plt.plot(X, true_function, color="black", label="True function")
plt.scatter(X, y, s=10, alpha=0.35, label="Observations")

# Predictive mean
plt.plot(X, pred_mean, color="tab:blue", label="Predictive mean")

# Uncertainty band (mean ± 2 std)
plt.fill_between(
    X,
    pred_mean - 2 * pred_std,
    pred_mean + 2 * pred_std,
    alpha=0.25,
    label="Uncertainty band (± 2σ)",
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Uncertainty Quantification (Simulated Predictive Distribution)")
plt.legend()
plt.tight_layout()
plt.show()

=======
"""
Uncertainty Quantification
==========================

This example demonstrates a simple form of uncertainty quantification (UQ)
for a regression-like setting.

We simulate a *predictive distribution* by sampling multiple model predictions
(e.g., think "ensemble" predictions). From these samples we compute:

- Predictive mean
- Predictive standard deviation
- An uncertainty band (mean ± 2σ)
"""

import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.linspace(0, 10, 100)
y_mean = np.sin(X)
y_std = 0.2 + 0.1 * np.abs(np.cos(X))

# Confidence interval
upper = y_mean + 2 * y_std
lower = y_mean - 2 * y_std

# Plot
plt.figure(figsize=(8, 4))
plt.plot(X, y_mean, label="Mean prediction")
plt.fill_between(X, lower, upper, alpha=0.3, label="95% confidence interval")

plt.legend()
plt.title("Predictive Uncertainty Quantification")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
>>>>>>> Stashed changes
