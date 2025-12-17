"""
Uncertainty Quantification
==========================

This example demonstrates how uncertainty representations can be converted
into numerical uncertainty measures such as variance and entropy.

We simulate an ensemble of probabilistic predictions and quantify uncertainty
across the ensemble.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

rng = np.random.default_rng(0)

# Simulated ensemble predictions for a binary classifier
n_models = 20
probs_class_1 = rng.beta(a=2, b=2, size=n_models)

# Variance-based uncertainty
variance = np.var(probs_class_1)

# Entropy-based uncertainty
mean_prob = np.mean(probs_class_1)
ent = entropy([mean_prob, 1 - mean_prob])

plt.figure(figsize=(5, 3))
plt.hist(probs_class_1, bins=10, edgecolor="black")
plt.axvline(mean_prob, linestyle="--", label="Mean prediction")
plt.title("Ensemble Predictions")
plt.xlabel("Predicted probability for class 1")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Variance-based uncertainty: {variance:.4f}")
print(f"Entropy-based uncertainty:  {ent:.4f}")
