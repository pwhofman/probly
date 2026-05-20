"""========================
Credal BNN Visualization
========================

This example trains a Bayesian ensemble using ``credal_bnn`` and renders
the resulting Sample-Mean Convex Credal Sets.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from probly.method.credal_bnn import credal_bnn
from probly.plot.credal import plot_credal_set
from probly.representer import representer
from examples.utils.model import MLPClassifier

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# %%
# 1. Prepare a 3-class dataset
# ----------------------------

# Use a wider equilateral triangle configuration so the classes are further away
centers = [[-7.0, -4.0], [0.0, 8.0], [7.0, -4.0]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=2.0, random_state=42)
X_train, X_test_data, y_train, y_test_data = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test_data).float()
y_test_tensor = torch.from_numpy(y_test_data).long()

# %%
# 2. Define a base deterministic model
# ------------------------------------

base_model = MLPClassifier(in_features=2, hidden_features=64, out_features=3)

# %%
# 3. Train base model and wrap with Credal BNN
# --------------------------------------------

# Convert base model to output probabilities before wrapping. This ensures
# the credal set vertices are computed correctly in probability space.
prob_model = nn.Sequential(base_model, nn.Softmax(dim=1))
credal_model = credal_bnn(prob_model, predictor_type="probabilistic_classifier", num_members=5)

for member in credal_model:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=0.01)
    for _ in range(2):
        opt.zero_grad()
        # member[0] is the base model which yields raw logits needed for cross_entropy
        logits = member[0](X_train_tensor)
        loss = nn.functional.cross_entropy(logits, y_train_tensor)
        loss.backward()
        opt.step()
    member.eval()

rep = representer(credal_model)
X_test = torch.tensor([
    [-7.0, -4.0],  # Near Class 0 (Center of blob 0)
    [0.0, 0.0],    # OOD Point (Equidistant from all classes)
    [0.0, 8.0],    # Near Class 1 (Center of blob 1)
])
credal_sets = rep.predict(X_test)

plot = plot_credal_set(
    credal_sets,
    title="Credal BNN",
    labels=["Class 0", "Class 1", "Class 2"],
    series_labels=["Near Class 0", "OOD Point", "Near Class 1"],
    show=True,
)
