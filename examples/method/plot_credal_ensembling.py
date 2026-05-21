"""===============================
Credal Ensembling Visualization
===============================

This example creates an ensemble of standard neural networks on a 3-class classification
problem using ``credal_ensembling``. We then visualize the predicted credal sets
(represented as a Convex Credal Set) for a few test points using a ternary simplex plot.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from probly.method.credal_ensembling import credal_ensembling
from probly.plot.credal import plot_credal_set
from probly.representer import representer
from examples.utils.model import MLPClassifier

np.random.seed(42)
torch.manual_seed(42)

# %%
# Prepare a 3-class dataset
# ----------------------------


centers = [[-7.0, -4.0], [0.0, 8.0], [7.0, -4.0]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=2.0, random_state=42)
X_train, X_test_data, y_train, y_test_data = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test_data).float()
y_test_tensor = torch.from_numpy(y_test_data).long()

# %%
# Define a base deterministic model
# ------------------------------------

base_model = MLPClassifier(in_features=2, hidden_features=64, out_features=3)

# %%
# Train base model and wrap with Credal Ensembling
# ---------------------------------------------------

prob_model = nn.Sequential(base_model, nn.Softmax(dim=1))
credal_model = credal_ensembling(prob_model, predictor_type="probabilistic_classifier", num_members=5)

for member in credal_model:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=0.01)
    for _ in range(1):
        opt.zero_grad()
        logits = member[0](X_train_tensor)
        loss = nn.functional.cross_entropy(logits, y_train_tensor)
        loss.backward()
        opt.step()
    member.eval()

# %%
# Predict and plot the credal sets
# -----------------------------------

rep = representer(credal_model)
X_test = torch.tensor([
    [-7.0, -4.0],
    [0.0, 0.0],
    [0.0, 8.0],
])
credal_sets = rep.predict(X_test)

plot = plot_credal_set(
    credal_sets,
    title="Credal Ensembling (Convex Set)",
    labels=["Class 0", "Class 1", "Class 2"],
    series_labels=["Near Class 0", "OOD Point", "Near Class 1"],
    show=True,
)
