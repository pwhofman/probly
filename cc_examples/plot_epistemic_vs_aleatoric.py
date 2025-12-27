"""
Epistemic vs Aleatoric Uncertainty
=================================

This is a minimal Sphinx-Gallery example.
"""

import numpy as np
import matplotlib.pyplot as plt


# %%
# Generate synthetic data
# -----------------------
#
# We create a simple regression problem with heteroscedastic noise
# (input-dependent aleatoric uncertainty).

rng = np.random.default_rng(42)

X = np.linspace(-3, 3, 200)
true_function = np.sin(X)

# Aleatoric uncertainty: noise level depends on x
aleatoric_std = 0.1 + 0.3 * (np.abs(X) / X.max())
y = true_function + rng.normal(0, aleatoric_std)


# %%
# Simulate epistemic uncertainty
# ------------------------------
#
# We simulate epistemic uncertainty by sampling multiple model predictions.
# In practice, this could come from ensembles, Bayesian neural networks,
# or Monte Carlo dropout.

n_models = 20
model_predictions = []

for _ in range(n_models):
    noise = rng.normal(0, 0.15, size=len(X))
    model_predictions.append(true_function + noise)

model_predictions = np.asarray(model_predictions)

epistemic_mean = model_predictions.mean(axis=0)
epistemic_std = model_predictions.std(axis=0)


# %%
# Plot epistemic vs aleatoric uncertainty
# ---------------------------------------

plt.figure(figsize=(8, 4))

plt.plot(X, true_function, color="black", label="True function")
plt.scatter(X, y, s=10, alpha=0.4, label="Observed data")

# Aleatoric uncertainty band
plt.fill_between(
    X,
    true_function - aleatoric_std,
    true_function + aleatoric_std,
    color="tab:blue",
    alpha=0.3,
    label="Aleatoric uncertainty",
)

# Epistemic uncertainty band
plt.fill_between(
    X,
    epistemic_mean - epistemic_std,
    epistemic_mean + epistemic_std,
    color="tab:orange",
    alpha=0.3,
    label="Epistemic uncertainty",
)

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Epistemic vs Aleatoric Uncertainty")
plt.tight_layout()
plt.show()


