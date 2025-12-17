"""
Selective Prediction and Overconfidence
=======================================

This example shows how quantified uncertainty can be used to reject uncertain
predictions and improve decision quality.

We simulate prediction confidence and accuracy and build an
accuracy–rejection curve.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

n_samples = 500

# Simulated confidences and correctness
confidence = rng.uniform(0, 1, size=n_samples)
correct = rng.random(n_samples) < confidence  # overconfident model

# Sort by confidence (low confidence rejected first)
order = np.argsort(confidence)
confidence_sorted = confidence[order]
correct_sorted = correct[order]

rejection_rates = np.linspace(0, 0.9, 30)
accuracies = []

for r in rejection_rates:
    k = int((1 - r) * n_samples)
    accuracies.append(correct_sorted[-k:].mean())

plt.figure(figsize=(5, 4))
plt.plot(rejection_rates, accuracies, marker="o")
plt.xlabel("Rejection rate")
plt.ylabel("Accuracy")
plt.title("Accuracy–Rejection Curve")
plt.grid(True)
plt.tight_layout()
plt.show()
