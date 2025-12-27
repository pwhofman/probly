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

